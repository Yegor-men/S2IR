def count_parameters(pytorch_model):
	total = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
	print(f"Trainable parameters: {total:,}")
