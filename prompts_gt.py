PROMPTS = {"prompt1": """
   You are a safety assistant for a quadcopter that is flying over a city and needs to perform an emergency landing on a surface. 
   The quadcopter will provide you with photos of several possible surfaces taken by its camera bellow, and ask questions of the suitability of the surfaces to perform the safest emergency landing possible.
   To classify as safe a potential landing site MUST take into consideration the following factors:


   - MUST be clear of any obstructions such as air ducts, cars, rubble etc.
   - MUST be clear of people 
   - MUST be a flat surface
   - You must return an answer

""",
"conversation-1":"""
    Rank the following surfaces from the most suitable to land the least one.
    First, explain your reasoning behind the surface ranking in detail, including the analysis for each one.\n 
    Then, output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.\n The index starts at 0.
""",
"conversation-2":"""
    This image represents closeup of a potential surface previously selected by yourself, please either confirm  whether this is a safe surface to land.
    First, explain your reasoning for deciding if the surface is suitable for a final approach or not
    Finally output either 1 to confirm or 0 to cancel the final approach.
""",

"original":"""
A quadcopter is flying over a city and needs to perform an emergency landing on a surface.

Given photos of several possible surfaces taken by this quadcopter, you are required to select the optimal surface as the emergency landing zone.

Avoid any obstructions or people when landing.\nFirst, explain your reasoning behind the surface ranking in detail, including the analysis for each one.

Then, output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.

The index starts at 0.
"""}

GROUND_TRUTH = {
    ## playerStart position scenario 1 (X=69789.929909,Y=-14527.701608,Z=158.775536)
    "scenario1": {
        "x_min": 534,
        "x_max": 714,
        "y_min": 119,
        "y_max": 430,
        "center_x":624,
        "center_y":275,
        "x_real":-122.49,
        "y_real": 19.96
    },
    # player Start position scenario 2 (X=36226.923474,Y=-24778.931072,Z=181.969621)
    "scenario2": {
        "x_min": 107,
        "x_max": 445,
        "y_min": 86,
        "y_max": 353,
        "center_x":276,
        "center_y":220,
        "x_real":53.65,
        "y_real": 55.92
    }
}