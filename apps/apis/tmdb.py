import os
import requests

from tqdm import tqdm
from dotenv import load_dotenv
from typing import Union, Iterable

from apps.utils.utils import _binary2image

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
    '''
    Get IMDB ID from TMDB ID
    '''
    url = self._get_content_url(content_id, content_type, '/external_ids')
    response_json: dict = self._request_query(url).json()
    return response_json.get('imdb_id', None)
  
  def get_tmdb_id(self, imdb_id: str, content_type: str='movie'):
    '''
    Get TMDB ID from IMDB ID
    '''
    url = f'https://api.themoviedb.org/3/find/{imdb_id}?\
      api_key={self.API_KEY}&external_source=imdb_id'
    response_json: dict = self._request_query(url).json()
    for result_type, result in response_json.items():
      if len(result) != 0:
        return result[0].get('id', None)
      
  def get_tmdb_ids(self, imdb_ids: Iterable, content_type: str='movie'):
    tmdb_ids = list()
    for imdb_id in tqdm(imdb_ids):
      tmdb_ids.append(self.get_tmdb_id(imdb_id, content_type))
    return tmdb_ids

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
  