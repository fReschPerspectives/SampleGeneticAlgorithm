class Capital:
    """
    This class is for initiating an object for each of the 50 US State Capitals;
    it includes the following characteristics: Name, State, Latitude, Longitude, Population
    """


    def __init__(self,
                 name: str,
                 state: str,
                 latitude: float,
                 longitude: float,
                 population: float,
                 ):
        self.Name = name
        self.State = state
        self.Latitude = latitude
        self.Longitude = longitude
        self.Population = population


    @staticmethod
    def create_state_capitals():
        capitals = [
            Capital("Montgomery", "Alabama", 32.3792, -86.3077, 200603),
            Capital("Juneau", "Alaska", 58.3019, -134.4197, 32255),
            Capital("Phoenix", "Arizona", 33.4484, -112.0740, 1608139),
            Capital("Little Rock", "Arkansas", 34.7465, -92.2896, 202591),
            Capital("Sacramento", "California", 38.5816, -121.4944, 524943),
            Capital("Denver", "Colorado", 39.7392, -104.9903, 727211),
            Capital("Hartford", "Connecticut", 41.7658, -72.6734, 121054),
            Capital("Dover", "Delaware", 39.1582, -75.5244, 39403),
            Capital("Tallahassee", "Florida", 30.4383, -84.2807, 196169),
            Capital("Atlanta", "Georgia", 33.7490, -84.3880, 498715),
            Capital("Honolulu", "Hawaii", 21.3069, -157.8583, 347397),
            Capital("Boise", "Idaho", 43.6150, -116.2023, 235684),
            Capital("Springfield", "Illinois", 39.7817, -89.6501, 114394),
            Capital("Indianapolis", "Indiana", 39.7684, -86.1581, 887642),
            Capital("Des Moines", "Iowa", 41.5868, -93.6250, 214133),
            Capital("Topeka", "Kansas", 39.0473, -95.6752, 126587),
            Capital("Frankfort", "Kentucky", 38.2009, -84.8733, 28602),
            Capital("Baton Rouge", "Louisiana", 30.4515, -91.1871, 227470),
            Capital("Augusta", "Maine", 44.3107, -69.7795, 18899),
            Capital("Annapolis", "Maryland", 38.9784, -76.4922, 40812),
            Capital("Boston", "Massachusetts", 42.3601, -71.0589, 675647),
            Capital("Lansing", "Michigan", 42.7325, -84.5555, 118210),
            Capital("Saint Paul", "Minnesota", 44.9537, -93.0900, 311527),
            Capital("Jackson", "Mississippi", 32.2988, -90.1848, 153701),
            Capital("Jefferson City", "Missouri", 38.5767, -92.1735, 42938),
            Capital("Helena", "Montana", 46.5891, -112.0391, 33124),
            Capital("Lincoln", "Nebraska", 40.8136, -96.7026, 289102),
            Capital("Carson City", "Nevada", 39.1638, -119.7674, 55916),
            Capital("Concord", "New Hampshire", 43.2081, -71.5376, 43976),
            Capital("Trenton", "New Jersey", 40.2206, -74.7597, 83387),
            Capital("Santa Fe", "New Mexico", 35.6870, -105.9378, 84683),
            Capital("Albany", "New York", 42.6526, -73.7562, 99224),
            Capital("Raleigh", "North Carolina", 35.7796, -78.6382, 467665),
            Capital("Bismarck", "North Dakota", 46.8083, -100.7837, 73529),
            Capital("Columbus", "Ohio", 39.9612, -82.9988, 898553),
            Capital("Oklahoma City", "Oklahoma", 35.4676, -97.5164, 681054),
            Capital("Salem", "Oregon", 44.9429, -123.0351, 174365),
            Capital("Harrisburg", "Pennsylvania", 40.2732, -76.8867, 49271),
            Capital("Providence", "Rhode Island", 41.8240, -71.4128, 190934),
            Capital("Columbia", "South Carolina", 34.0007, -81.0348, 137300),
            Capital("Pierre", "South Dakota", 44.3683, -100.3510, 13961),
            Capital("Nashville", "Tennessee", 36.1627, -86.7816, 689447),
            Capital("Austin", "Texas", 30.2672, -97.7431, 961855),
            Capital("Salt Lake City", "Utah", 40.7608, -111.8910, 199723),
            Capital("Montpelier", "Vermont", 44.2601, -72.5754, 7855),
            Capital("Richmond", "Virginia", 37.5407, -77.4360, 226622),
            Capital("Olympia", "Washington", 47.0379, -122.9007, 55605),
            Capital("Charleston", "West Virginia", 38.3498, -81.6326, 48864),
            Capital("Madison", "Wisconsin", 43.0731, -89.4012, 258366),
            Capital("Cheyenne", "Wyoming", 41.1400, -104.8202, 65132)
        ]
        return capitals