
class DotDict:
    def __init__(self, dictionary):
        self.inner_dict={}
        self.append(dictionary)

    def append(self,dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, DotDict(value))
                else:
                    setattr(self, key, value)
                    self.inner_dict[key] = value