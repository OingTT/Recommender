import unittest
from PIL import WebPImagePlugin

from apps.apis.just_watch import JustWatch

class Test_JustWatch(unittest.TestCase):
  
  john_wick_title = 'John Wick'
  john_wick_just_watch_id = 153649
  john_wick_tmdb_id = 245891
  
  def setUp(self):
    self.just_watch = JustWatch()

  def test_search_for_item(self):
    param = {
      'query': self.john_wick_title
    }
    result = self.just_watch._search_for_item(**param)
    assert len(result['items']) != 0, ''
  
  def test_get_tmdb_id(self):
    param = {
      'query': self.john_wick_title
    }
    item = self.just_watch._search_for_item(**param)['items'][0]
    tmdb_id = self.just_watch.get_tmdb_id(item)
    assert tmdb_id == self.john_wick_tmdb_id, ''

  def test_search_by_providers(self):
    result = self.just_watch.search_by_providers(['nfx'])['items']
    assert len(result) != 0, ''

  def test_get_just_watch_image(self):
    url_path = 'poster/187285619/s592/john-wick.webp'
    result = self.just_watch.get_just_watch_image(url_path)
    assert isinstance(result, WebPImagePlugin.WebPImageFile), '{}'.format(type(result))
  
  def test_get_available_providers(self):
    result = self.just_watch.get_available_providers()
    assert len(result) != 0, ''
    
  def test_available_providers_parser(self):
    providers = self.just_watch.get_available_providers()
    self.just_watch._available_providers_parser(providers)
    
