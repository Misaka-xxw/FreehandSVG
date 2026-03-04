from core.basic import ImageNp

class BasicFilter:
    def __init__(self, image: ImageNp):
        self.image = image

    def apply(self) -> ImageNp:
        raise NotImplementedError("Subclasses must implement the apply method.")
