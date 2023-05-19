import unittest
from apps.apis.TMDB import TMDB
from PIL import JpegImagePlugin

class Test_TMDB(unittest.TestCase):

  john_wick_title = 'John Wick'
  john_wick_tmdb_id = 245891
  john_wick_imdb_id = 'tt2911666'

  def setUp(self):
    self.tmdb = TMDB()
    
  def test_get_content_url(self):
    result = self.tmdb._get_content_url(self.john_wick_tmdb_id)
    assert type(result) is str and len(result) != 0, ''

  def test_request_query(self):
    url = self.tmdb.BASE_URL + f'movie/{self.john_wick_tmdb_id}'
    response = self.tmdb._request_query(url)
    assert response.status_code == 200, ''

  def test_get_imdb_id(self):
    response = self.tmdb.get_imdb_id(self.john_wick_tmdb_id)
    
    assert response == self.john_wick_imdb_id, ''

  def test_get_tmdb_id(self):
    response = self.tmdb.get_tmdb_id(self.john_wick_imdb_id)
    
    assert response == self.john_wick_tmdb_id, ''

  def test_search_movie(self):
    response = self.tmdb.search_movie(self.john_wick_title)

    assert len(response['results']) != 0\
      , '{}'.format(response['results'])

  def test_get_content_detail(self):
    response = self.tmdb.get_content_detail(self.john_wick_tmdb_id)
    assert response['id'] == self.john_wick_tmdb_id, ''

  def test_get_watch_providers(self):
    response = self.tmdb.get_watch_providers(self.john_wick_tmdb_id)
    assert response['id'] == self.john_wick_tmdb_id, ''

  def test_get_tmdb_image(self):
    response = self.tmdb.get_tmdb_image('/tbEdFQDwx5LEVr8WpSeXQSIirVq.jpg')
    assert isinstance(response, JpegImagePlugin.JpegImageFile), ''
  
  def test_get_recommendations(self):
    response = self.tmdb.get_recommendations(self.john_wick_tmdb_id)
    assert response['page'] == 1, ''