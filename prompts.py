PROMPTS = {"prompt1": """
   You are a safety asisstant for a quadcopter that is flying over a city and needs to perform an emergency landing on a surface. 
   Given photos of several possible surfaces taken by this quadcopter, you are required to select the safest surface to perform the emergency landing.
   To classify as safe a potential landing site MUST take into consideration the following factors:


   - MUST be clear of any obstructions such as air ducts, cars, rubble etc.
   - MUST be clear of people 
   - MUST be a flat surface
   
           
   First, explain your reasoning behind the surface ranking in detail, including the analysis for each one.\n 
   Then, output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.\n The index starts at 0.",

""",



"original":"""
A quadcopter is flying over a city and needs to perform an emergency landing on a surface.

Given photos of several possible surfaces taken by this quadcopter, you are required to select the optimal surface as the emergency landing zone.

Avoid any obstructions or people when landing.\nFirst, explain your reasoning behind the surface ranking in detail, including the analysis for each one.

Then, output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.

The index starts at 0.
"""}