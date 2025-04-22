import torch
from gat_model import GAT
from ts.torch_handler.base_handler import BaseHandler

class GATHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        self.model = GAT(in_channels=124)  # replace 124 with your feature count
        self.model.load_state_dict(torch.load(f"{model_dir}/gnn_lstm_model.pt"))
        self.model.eval()

    def handle(self, data, context):
        try:
            input_tensor = torch.tensor(data[0]["body"], dtype=torch.float)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
            return output.tolist()
        except Exception as e:
            return [f"❌ Error: {str(e)}"]

