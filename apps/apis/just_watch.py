import requests
from justwatch import JustWatch as _JustWatch

from ..utils.utils import _binary2image

class JustWatch:

  def __init__(self, country='KR'):
    self.just_watch = _JustWatch(country=country)

  def _search_for_item(self, **kargs: dict) -> dict:
    param = dict()
    for key, value in kargs.items():
      if value is not None:
        param.update( { key : value } )
    result = self.just_watch.search_for_item(**param)
    return result

  def get_tmdb_id(self, item: dict) -> int:
    '''
    return tmdb id if exist or -1
    '''
    scorings = item['scoring']
    for scoring in scorings:
      if scoring['provider_type'] == 'tmdb:id':
        return scoring['value']
    return -1

  def search_by_providers(self, providers: list=None, content_types: list=['movie'], monetization_types: list=None):
    param = {'providers': providers, 'content_types': content_types, 'monetization_types': monetization_types}
    result = self._search_for_item(**param)
    if len(result['items']) == 0:
      raise Exception
    return result

  def get_just_watch_image(self, url_path: str):
    image_url = f'https://images.justwatch.com/{url_path}'
    response = requests.get(image_url)
    response.raise_for_status()
    image = _binary2image(response.content)
    return image

  def get_available_providers(self):
    result = self.just_watch.get_providers()
    return result
  
  def _available_providers_parser(self, providers):
    for provider in providers:
      s = ''
      for k, v in provider.items():
        s += f'{k}: {v}, '

j = JustWatch()
a = j.get_available_providers()
print(a)
# j._available_providers_parser(a)