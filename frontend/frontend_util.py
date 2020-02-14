def get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def identity():
    def _impl(inputs, input_types):
        return inputs[0]
    return _impl
