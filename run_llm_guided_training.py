# MNIST + GPT-4o-guided weight/hparam modulation system
# Includes: Self-evaluation loop, ΔW blending, JSON correction, visualization, and model saving

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json
import openai
import os
import matplotlib.pyplot as plt
import re

# ✅ Set your OpenAI API key (use environment variable or insert directly here)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 1. Load data (only 10 random samples for few-shot learning)
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    small_indices = torch.randperm(len(full_train))[:10]  # Use only 10 samples from 60,000
    small_train = Subset(full_train, small_indices)

    test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
    train_loader = DataLoader(small_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    return train_loader, test_loader

# ✅ 2. Define CNN model
class MNISTModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )
        self.dropout_p = dropout

    def forward(self, x):
        return self.fc(self.conv(x))

    def set_dropout(self, p):
        # Dynamically adjust dropout during training
        for layer in self.fc:
            if isinstance(layer, nn.Dropout):
                layer.p = p
        self.dropout_p = p

# ✅ 3. Evaluate model on given dataloader
def evaluate(model, dataloader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {'acc': correct / total, 'loss': loss_sum / total}

# ✅ 4. LLM prompt builder for tuning parameters
def build_prompt(epoch, train_acc, val_acc, dropout, lr):
    return f"""
You are an AI assistant helping tune a machine learning model.

Epoch {epoch}:
- Train Accuracy: {train_acc:.4f}
- Validation Accuracy: {val_acc:.4f}
- Dropout: {dropout:.2f}
- Learning Rate: {lr:.6f}

Carefully evaluate the accuracy.
- Adjust `dropout` by ±0.05 if not perfect.
- Use non-zero alpha in range 0.2–0.6 unless accuracy dropped.
- Modify learning rate if accuracy plateaued.

Respond ONLY with raw valid JSON, no explanation:
{{
  "dropout": float,
  "learning_rate": float,
  "alpha": float,
  "weight_modulation": {{
    "parameter_name": list of floats or nested lists
  }}
}}
"""

# ✅ 5. Self-reflection prompt after each epoch
def build_self_eval_prompt(prev_val_acc, current_val_acc, prev_adjustment):
    return f"""
Previous validation accuracy: {prev_val_acc:.4f}
Current validation accuracy: {current_val_acc:.4f}
Adjustment:
Dropout: {prev_adjustment['dropout']}
Learning Rate: {prev_adjustment['learning_rate']}
Alpha: {prev_adjustment['alpha']}

Was the adjustment beneficial? Respond in 1 sentence only.
"""

# ✅ 6. Call GPT-4o with a formatted prompt
def gpt4o_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI model adjustment assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content']

# ✅ 7. Main training loop with LLM modulation
def train_model():
    train_loader, test_loader = get_dataloaders()
    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(10):
        # Standard training step
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        # Evaluate model
        train_metrics = evaluate(model, train_loader)
        val_metrics = evaluate(model, test_loader)

        # Prompt GPT-4o for guidance
        prompt = build_prompt(epoch, train_metrics['acc'], val_metrics['acc'], model.dropout_p, optimizer.param_groups[0]['lr'])
        raw_response = gpt4o_call(prompt)

        # Try parsing JSON from response
        try:
            match = re.search(r'{.*}', raw_response, re.DOTALL)
            if match:
                adjustment = json.loads(match.group(0))
            else:
                raise ValueError("No JSON found.")
        except Exception as e:
            print("[Warning] Failed to parse JSON. Response was:\n", raw_response)
            adjustment = {"dropout": 0.3, "learning_rate": 0.0007, "alpha": 0.4, "weight_modulation": {}}

        # Apply adjustment: blended weight update
        alpha = adjustment['alpha']
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_update = -optimizer.param_groups[0]['lr'] * param.grad
                delta = adjustment.get("weight_modulation", {}).get(name, None)
                if delta:
                    delta_tensor = torch.tensor(delta, dtype=param.dtype, device=param.device).view_as(param.data)
                    param.data += (1 - alpha) * grad_update + alpha * delta_tensor
                else:
                    param.data += (1 - alpha) * grad_update

        # Apply new dropout and learning rate
        model.set_dropout(adjustment['dropout'])
        optimizer.param_groups[0]['lr'] = adjustment['learning_rate']

        # Prompt GPT-4o for self-evaluation
        feedback = ""
        if epoch > 0:
            prev_val_acc = history[-1]['val']['acc']
            feedback_prompt = build_self_eval_prompt(prev_val_acc, val_metrics['acc'], history[-1]['adjustment'])
            feedback = gpt4o_call(feedback_prompt)

        print(f"Epoch {epoch}: Train Acc = {train_metrics['acc']:.4f}, Val Acc = {val_metrics['acc']:.4f}, α = {alpha}")
        if feedback:
            print("Self-Eval:", feedback)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "adjustment": adjustment,
            "self_eval": feedback
        })

    # ✅ Save model
    torch.save(model.state_dict(), "mnist_llm_model.pt")

    # ✅ Plot validation accuracy, alpha, and dropout changes
    epochs = [h['epoch'] for h in history]
    val_accs = [h['val']['acc'] for h in history]
    alphas = [h['adjustment']['alpha'] for h in history]
    dropouts = [h['adjustment']['dropout'] for h in history]

    plt.plot(epochs, val_accs, label="Val Acc")
    plt.plot(epochs, alphas, label="Alpha")
    plt.plot(epochs, dropouts, label="Dropout")
    plt.xlabel("Epoch")
    plt.title("Training Dynamics (LLM-guided)")
    plt.legend()
    plt.grid(True)
    plt.savefig("mnist_llm_results.png")
    plt.show()

if __name__ == '__main__':
    train_model()
