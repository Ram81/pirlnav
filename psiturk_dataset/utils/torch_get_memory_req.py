import torch


def get_tensor_memory_in_gb(shape=(480, 640, 3)):
    data = torch.zeros(shape)
    memory_in_bytes = data.float().element_size() * data.nelement()
    return memory_in_bytes / (1024**3)


def get_total_input_memory(num=500, batch_size=8):
    rgb = get_tensor_memory_in_gb((num, batch_size, 480, 640, 3))
    depth = get_tensor_memory_in_gb((num, batch_size, 480, 640, 1))
    semantic = get_tensor_memory_in_gb((num, batch_size, 480, 640, 1))
    pred_semantic = get_tensor_memory_in_gb((num, batch_size, 480, 640, 1))
    actions = get_tensor_memory_in_gb((num, batch_size))
    instructions = get_tensor_memory_in_gb((num, batch_size, 256))
    print("RGB observations: {}".format(rgb))
    print("Depth observations: {}".format(depth))
    print("Semantic observations: {}".format(semantic))
    print("Pred observations: {}".format(pred_semantic))
    print("Actions: {}".format(actions))
    print("Instructions: {}".format(instructions))
    print("Total memory requirement: {}".format(rgb + depth + semantic + pred_semantic + actions * 2 + instructions))


if __name__ == "__main__":
    get_total_input_memory()
