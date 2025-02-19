import json
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)

# https://www.ncdc.noaa.gov/cdo-web/webservices/v2

MAX_RETRIES = 5

class NOAAController:
    def __init__(self, token):
        self.token = token
        self.climate_api_url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/'
        self.storm_api_url = 'https://www.ncdc.noaa.gov/stormevents/api/v1/'
    
    """

    each token will be limited to five requests per second (429) and 10,000 requests per day
    
    """

    def fetch_data(self, endpoint, params = dict()):
        next_retry_time = 1
        next_retry_delta = 2
        retry = 0
        good_response = True
        while good_response and retry < MAX_RETRIES:
            headers = {'token': self.token}

            url = self.climate_api_url + endpoint

            response = requests.get(url, headers=headers, params=params) # TODO: allow passing of parameters into data

            # logging.info(f"Response code: {response.status_code}")
            response.close()
            match(response.status_code):
                case 200: # accepted
                    # the response's data is JSON
                    response_data = response.json()
                    return response_data
                case 429: 
                    logging.info(f"NOAA request limit reached {retry+1}. Retrying in {next_retry_time} seconds")
                    time.sleep(next_retry_time)
                    next_retry_time += next_retry_delta
                    break
                case _:
                    logging.info(f"NOAA API Error: {response.status_code} - {response.text}")
                    good_response = False
                    break
            retry += 1
        return {}

    def climate_query(self, endpoint):
        pass

    def storm_query(self):

        # params = {
        #     "startDate": "2023-01-01",   # Start date (YYYY-MM-DD)
        #     "endDate": "2023-12-31",     # End date (YYYY-MM-DD)
        #     "eventType": "Flood",      # Weather event type (e.g., Tornado, Hurricane, Flood)
        #     # "fatalities__gte": 1,        # Only events with fatalities
        #     # "format": "json",            # Response format
        #     "limit": 10                  # Limit results for testing
        # }

        response = requests.get(self.storm_api_url + 'events') # params=params

        response.close()

        if response.status_code == 200:
            data = response.json()

            print(data)
        else:
            print(f"Error: {response.status_code} - {response.text}")


            # url = "https://www.ncdc.noaa.gov/stormevents/ftp.jsp"  # Replace with the actual download link

            # response = requests.get(url)

            

            # with open("storm_events.csv", "wb") as file:

            #     file.write(response.content)
