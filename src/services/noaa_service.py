from src.controllers.noaa_controller import NOAAController
from src.utility import multiget

"""
Specify time period
Specify region(s) (?)

"""

class NOAAService:
    def __init__(self, token):
        self.controller = NOAAController(token=token)

    # single
    def retrieve_station_data(self, stationid: str, limit: int, offset: int = 1):
        response = self.controller.fetch_data(endpoint=f'data', 
                                              params={'stationid': stationid, 
                                                      'limit': limit,
                                                      'offset': offset})

        return response


    # full
    def retrieve_all_station_data(self, stationid: str):
        all_data = []
        limit_window = 1000
        response = self.retrieve_station_data(stationid=stationid, limit=limit_window)
        all_data.extend(response.get('results'))
        # count = response.get('metadata', dict()).get('resultset', dict()).get('count', 0)
        count = multiget(response, ['metadata', 'resultset', 'count'], 0)
        for i in range(1+limit_window, min(100, count), limit_window):
            all_data.extend(response.get('results'))

            response = self.retrieve_station_data(stationid=stationid, limit=limit_window, offset=i)

        return all_data


