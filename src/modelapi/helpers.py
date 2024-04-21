class IdConverter():
    def __init__(self) -> None:
        self._forward_conversation = dict()
        self._backward_conversation = list()

    def add(self, id):
        if id not in self._forward_conversation:
            self._forward_conversation[id] = self.get_count()
            self._backward_conversation.append(id)
        return self._forward_conversation[id]

    def get_forward(self, id):
        return self._forward_conversation.get(id, None)
    
    def get_backward(self, id):
        if id < 0 or id > self.get_count():
            return None
        return self._backward_conversation[id]

    def get_count(self):
        return len(self._backward_conversation)
