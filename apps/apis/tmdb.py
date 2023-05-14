import os
import requests

from dotenv import load_dotenv
from typing import Union

from ..utils.utils import _binary2image

class TMDB:
  BASE_URL = 'https://api.themoviedb.org/3/'
  CONTENT_URL = 'https://api.themoviedb.org/3/{content_type}/{content_id}'

  def __init__(self):
    load_dotenv()
    self.API_KEY = os.getenv('TMDB_API_KEY')

  def _get_content_url(self, content_id: str, content_type: str='movie', path: str=''):
    return self.CONTENT_URL.format(content_type=content_type, content_id=content_id) + path
  
  def _request_query(self, URL: str, **kargs: dict) -> requests.Response:
    param = dict( api_key = self.API_KEY )
    for key, value in kargs.items():
      if value is not None:
        param.update( { key : value } )
    response = requests.get(URL, params=param)
    response.raise_for_status()

    return response

  def get_imdb_id(self, content_id: str, content_type: str='movie') -> Union[str, None]:
    url = self._get_content_url(content_id, content_type, '/external_ids')
    response_json: dict = self._request_query(url).json()
    return response_json.get('imdb_id', None)

  def search_movie(self,
                  query: str,
                  language: str='ko_KR',
                  page: int=1,
                  include_adult: bool=True,
                  region: str='KR',
                  year: int=None,
                  primary_release_year: int=None):
    param = dict(
      query=query,
      language=language,
      page=page,
      include_adult=include_adult,
      region=region,
      year=year,
      primary_release_year=primary_release_year
    )
    url = self.BASE_URL + 'search/movie'
    response = self._request_query(url, **param)    
    return response.json()

  def get_content_detail(self, content_id: str, content_type: str='movie', language: str='ko_KR', append_to_response: str=None):
    url = self._get_content_url(content_id, content_type)
    response = self._request_query(url, **dict(
      language=language,
      append_to_response=append_to_response
    ))
    return response.json()
  
  def get_watch_providers(self, content_id: str, content_type: str='movie'):
    url = self._get_content_url(content_id, content_type, '/watch/providers')
    response = self._request_query(url)
    return response.json()

  def get_tmdb_image(self, url_path: str):
    '''
    get poster or backdrop image
    '''
    image_url = f'https://image.tmdb.org/t/p/original/{url_path}'
    response = requests.get(image_url)
    response.raise_for_status()
    image = _binary2image(response.content)
    
    return image

  def get_recommendations(self, content_id: str, content_type: str='movie', language: str='ko_KR', page: int=1):
    '''
    min page=1, max page=1000
    '''
    url = self._get_content_url(content_id, content_type, '/recommendations')
    response = self._request_query(url)
    return response.json()
