import torch 
import torch.nn as nn
import torch.utils.data as data
import sklearn.datasets as datasets
import sklearn.preprocessing as preprocessing

iris = datasets.load_iris()
X, y = iris.data, iris.target

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = data.TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False)
class SimpleNN(torch.nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x): 
        return self.model(x)
model = SimpleNN(input_size=4, hidden_size=16, output_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100 
for epoch in range(epochs): 
    model.train()
    for X_batch, y_batch in train_loader: 
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0: 
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    model.eval()
    correct, total = 0, 0 
    with torch.no_grad(): 
        for X_val, y_val in test_loader: 
            outputs = model(X_val)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)
    accuracy = correct / total
    if epoch % 10 == 0: 
        print(f"Validation Accuracy after epoch {epoch+1}: {accuracy:.4f}")

with torch.no_grad(): 
    correct, total = 0, 0
    for X_val, y_val in test_loader: 
        outputs = model(X_val)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds ==y_val).sum().item()
        total += y_val.size(0)
    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy:.4f}")
