import tensorflow as tf

class MyBox:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

    def get_min(self):
        return self.min_embed

    def get_delta(self):
        return self.delta_embed

    def get_max(self):
        return self.max_embed

    def get_log_delta(self):
        return tf.log(self.delta_embed)