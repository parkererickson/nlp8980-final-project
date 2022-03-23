# %%
import cl_graph_bert as cgm
import torch
from torch import nn
import json
from transformers import BertModel, BertConfig, BertTokenizer, AdamW
import tqdm

from torch.utils.data import Dataset, DataLoader

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels, ids):
        self.tokens = tokens
        self.labels = labels
        self.ids = ids
        
    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokens.items()}
        out = {"tokens": item, "label": self.labels[idx], "id": self.ids[idx]}
        return out

def process_text(filepath, batch_size):
    reviews = []
    data = open(filepath)
    for line in data.readlines():
        reviews.append(json.loads(line))

    review_texts = []
    review_scores = []

    for sample in reviews:
        if 'reviewText' in sample and 'overall' in sample:
            review_texts.append(sample['reviewText'])
            if sample['overall'] >= 4:
                review_scores.append(1)
            else:
                review_scores.append(0)
                
    train_reviews = review_texts[:len(review_texts)//2]
    train_ids = [i for i in range(0, len(review_texts)//2+1)]
    test_reviews = review_texts[len(review_texts)//2:]
    test_ids = [i for i in range(len(review_texts)//2, len(review_texts))]
    train_scores = review_scores[:len(review_texts)//2]
    test_scores = review_scores[len(review_texts)//2:]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenized_train_reviews = tokenizer(train_reviews, return_tensors="pt", padding='max_length', truncation=True)
    tokenized_test_reviews = tokenizer(test_reviews, return_tensors="pt", padding='max_length', truncation=True)

    train_dataset = ReviewDataset(tokenized_train_reviews, train_scores, train_ids)
    test_dataset = ReviewDataset(tokenized_test_reviews, test_scores, test_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

# %%
batch_size = 16
train_loader, test_loader = process_text("../review_dataset/Industrial_and_Scientific_5.json", batch_size)

# %%
import dgl
g = dgl.load_graphs("./graphs/industrial_and_scientific_5_core.dgl")[0][0]

# %%
model = cgm.CLIPGraphModel(
    rel_types = g.etypes,
    emb_types = {x: g.number_of_nodes(x) for x in g.ntypes} 
)

# %%
device = 'cuda'
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{"params":model.language_model.parameters(),"lr": 0.00001},
                              {"params":model.graph_model.parameters(), "lr": 0.001},
                              {"params":model.language_projection.parameters(), "lr": 0.001},
                              {"params":model.graph_projection.parameters(), "lr": 0.001}])

# %%
def tokens_to_cuda(tokens, device):
    dictionary = {}
    for key, value in tokens.items():
        dictionary[key] = value.to(device)
    return dictionary

# %%
epochs = 2

model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        
        tokens = tokens_to_cuda(batch['tokens'], device)
        loss = model(g.to(device), "Review", tokens, batch["id"].to(device))["loss"]    
                
        epoch_loss += loss
        
        loss.backward()
        optimizer.step()
       
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            tokens = tokens_to_cuda(batch['tokens'], device)
            loss = model(g.to(device), "Review", tokens, batch["id"].to(device))["loss"]

            val_loss += loss
        
    print("End of Epoch", epoch)
    print("Training loss:", epoch_loss)
    print("Test loss:", val_loss)

# %%
class BertGraphMLP(nn.Module):
    def __init__(self, model, encoder="language_emb", num_labels=2, finetune=False):
        super(BertGraphMLP, self).__init__()
        self.num_labels = num_labels
        self.BertGraph = model
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, g, tokens, ids):
        if finetune:
            out = self.BertGraph(g, "Review", tokens, ids)
        else:
            with torch.no_grad():
                out = self.BertGraph(g, "Review", tokens, ids)
        if self.encoder == "language_emb":
            embs = out["language_emb"]
        elif self.encoder == "graph_emb":
            embs = out["graph_emb"]
        elif self.encoder == "mean":
            embs = (out["langauge_emb"] + out["graph_emb"])/2
        else:
            raise("Not Implemented")
        logits = self.classifier(embs)
        return logits


# %%
import torch.nn.functional as F

def evaluate_review_sentiment(model, encoder, finetune=False):
    eval_model = BertGraphMLP(model, encoder, finetune=finetune)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params":eval_model.classifier.parameters(), "lr": 0.001},       
    ])

    epochs = 10
    device= 'cuda'
    eval_model.to(device)
    if not(finetune):
        eval_model.BertGraph.eval()
    preds = []

    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()  
            tokens = tokens_to_cuda(batch['tokens'], device)
            embs = eval_model(g.to(device), tokens, batch["id"].to(device))  
            loss = loss_function(embs, batch['label'].to(device)) / batch_size
            epoch_loss += loss
            loss.backward()
            optimizer.step()    
        print(epoch_loss/len(train_loader))

    eval_model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            tokens = tokens_to_cuda(batch['tokens'], device)
            scores = F.softmax(eval_model(g.to(device), tokens, batch["id"].to(device)))
            preds.extend(torch.round(F.softmax(scores)[:,1]).to(torch.int64))

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    preds_cpu = [i.to('cpu') for i in preds]
    print("End of training f1 score accuracy", f1_score(test_scores, preds_cpu))
    print("End of training accuracy", accuracy_score(test_scores, preds_cpu))
    print("End of training precision", precision_score(test_scores, preds_cpu))
    print("End of training recall", recall_score(test_scores, preds_cpu))

# %%
enc_methods = ["language_emb", "graph_emb", "mean"]

for method in enc_methods:
    print("====== EVALUATING:", method, "======")
    evaluate_review_sentiment(model, method)

# %%
torch.save(model.state_dict(), "./base_statedict_{}.pt".format(float(val_loss)))


