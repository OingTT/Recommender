class JustWatchProviders:
  pass

class JustWatchProvider:
  def __init__(self, id, technical_name, short_name, clear_name, monetization_types, slug):
    self.id = id
    self.technical_name = technical_name
    self.short_name = short_name
    self.clear_name = clear_name
    self.monetization_types = monetization_types
    self.slug = slug