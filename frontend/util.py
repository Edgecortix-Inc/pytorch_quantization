def get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()
