PROMPTS = {"prompt1": """
# Safe Drone Landing Asistant
   You are a safety assistant for a quadcopter that is flying over a city and needs to perform an emergency landing on a surface. 
   The quadcopter will provide you with photos of several possible surfaces taken by its camera bellow, and ask questions of the suitability of the surfaces to perform the safest emergency landing possible.
   To classify as safe a potential landing site MUST take into consideration the following factors:

## Constraints           

    - **MUST** be clear of any obstructions such as air ducts, cars, rubble etc.
    - **MUST** be clear of people 
    - **MUST** be a flat surface

## Clarifications 
           
    - Ignore non-critical visual features such as shadows, surface textures, tile patterns, or color variations unless they **CLEARLY** indicate an actual obstruction.
    - Do not assume a surface is unsafe just because it might be something like a shingled roof unless there is strong visual evidence of danger (e.g., visible slope, fragile material, obvious gaps).
    - If no surface is perfectly safe, select the one with the lowest risk.
    - You **MUST** always return a clear selection (never refuse).      

""",
"conversation-1":"""
    You are receiving up to **5 images** from the drone:

    - **Images 0–4** (maximum 5 images): Candidate surfaces for landing. 

    ### Task
    1. Analyze each candidate surface (Images 0–4).  
    2. Rank the candidate surfaces from **most suitable** to **least suitable** for landing.  
    3. Provide detailed reasoning for each candidate, explaining why it is more or less suitable. 
    4. Output the ranking as a list of indices (corresponding to the candidate images only).  

    ### Output Format
    1. **Detailed Explanation** (paragraphs describing reasoning for each candidate).  
    2. **Final Ranking**: output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.\n The index starts at 0. 

""",
"conversation-2":"""
    This image represents closeup of a potential surface previously selected by yourself, please either confirm  whether this is a safe surface to land.
    First, explain your reasoning for deciding if the surface is suitable for a final approach or not
    Finally output either 1 in the indices array to confirm or 0 to cancel the final approach.
""",
"conversation-2-1":"""
    You are receiving a **close-up image** of a potential landing surface that was previously selected by you.  
    Your task is to confirm whether this surface is safe for a **final landing approach**.

    ### Task
    1. Analyze the surface in the close-up image.  
    2. Decide if the surface is **safe to land** or **unsafe**.  
    3. Provide a detailed explanation of your reasoning.  
    4. Output your decision as an index array with a single value:  
        - 1 in the indices array to confirm or 0 to cancel the final approach.

    ### Notes
    - Ignore **non-critical details** such as shadows, lighting variations, or surface patterns unless they clearly indicate a real hazard.  
    - Confirm landing if the surface appears **flat, unobstructed, and free of people**.  
    - Deny landing only if there is clear evidence of danger (obstructions, slope, rubble, fragile/unsafe material).  
    - Always return a decision (never leave it blank).  

    ### Output Format
    1. **Explanation** (paragraph justifying your decision).  
    2. **Final Decision** in the form: 1 in the indices array to confirm or 0 to cancel the final approach.
""",
"conversation-1-2":"""
    You are receiving up to **6 images** from the drone:

    - **Images 0–4** (maximum 5 images): Candidate surfaces for landing.  
    - **Last image, marked as **Context Image** (always the final one)**: The full scene from the drone, provided **only for context**.  
    - Do **not** treat the last image as a landing candidate.  
    - Use it only to better understand the candidate surfaces.

    ### Task
    1. Analyze each candidate surface (Images 0–4).  
    2. Rank the candidate surfaces from **most suitable** to **least suitable** for landing.  
    3. Provide detailed reasoning for each candidate, explaining why it is more or less suitable.  
    4. Output the ranking as a list of indices (corresponding to the candidate images only).  

    ### Output Format
    1. **Detailed Explanation** (paragraphs describing reasoning for each candidate).  
    2. **Final Ranking**: output the indices corresponding to each photo, representing the ranking from the most suitable place to land to the least one.\n The index starts at 0. 
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
        "y_real": 19.96,
        "x_min_w": 210,
        "x_max_w": 392,
        "y_min_w": 148,
        "y_max_w": 421,
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
        "y_real": 55.92,
        "x_min_w": 615,
        "x_max_w": 746,
        "y_min_w": 71,
        "y_max_w": 360,
    }
}
OTHER_SURFACES_1 = {
    "x_min1": 457,
    "x_max1": 625,
    "y_min1": 507,
    "y_max1": 538,

    "x_min2": 661,
    "x_max2": 953,
    "y_min2": 0,
    "y_max2": 97,

    "x_min3": 1,
    "x_max3": 189,
    "y_min3": 384,
    "y_max3": 503,

}

OTHER_SURFACES_2 = {
    "x_min1": 287,
    "x_max1": 417,
    "y_min1": 503,
    "y_max1": 536,

    "x_min2": 819,
    "x_max2": 953,
    "y_min2": 121,
    "y_max2": 261,

    "x_min3": 475,
    "x_max3": 558,
    "y_min3": 37,
    "y_max3": 104,
    
}


ENVELOPE = {
  "task": "Pick a single safe rooftop landing zone for a quadcopter.",
  "input": {
    "description": "One top-down view image with green boxes labeled 0..N-1 indicating candidate landing areas",
    "candidates": {"count": "<N>", "labels": "0..N-1", "note": "Use the image only."}
  },
  "format": {
    "description": "Return ONLY JSON. No prose, no markdown.",
    "schema": {
      "oneOf": [
        {"type": "object", "required": ["reject","index","reason"],
         "properties": {"reject": {"const": False}, "index": {"type": "integer","minimum": 0},
                        "reason": {"type":"string"},
                        "ranking": {"type": "array", "items": {"type":"integer","minimum":0}}}}
      ],
      "additionalProperties": False
    }
  },
  "constraints": [
    "Avoid obstacles, parapet edges, vents, people, vehicles, and strong glare/shadow.",
    "Prefer boxes that look larger and more open in the image.",
  ],
  "examples": [
    {
      "input": "Image shows 4 boxes: 0..3.",
      "output": {"reject": False, "index": 0, "reason": "Largest open roof patch away from edges.", "ranking": [0,1,3,2]}
    }
  ]
}
