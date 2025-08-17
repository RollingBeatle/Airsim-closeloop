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
""",

"json_schema": {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "string",
                "format": "uri"
            }
        }
    },
    "required": ["images"]
}

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



ResponseFormatDecision = {
    "type": "json_schema",
    "json_schema": {
        "name": "landing_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reject": {"type": "boolean", "description": "True if no candidate is safe."},
                "index": {"type": "integer", "minimum": 0, "description": "Chosen candidate index if reject is false."},
                "ranking": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "description": "Optional ranking best→worst."
                },
                "reason": {"type": "string", "maxLength": 400},
                "reject_reason": {"type": "string", "maxLength": 400}
            },
            "required": ["reject"],
            "oneOf": [
                { "properties": { "reject": { "const": True },  "reject_reason": { "type": "string", "minLength": 1 } },
                  "required": ["reject_reason"] },
                { "properties": { "reject": { "const": False }, "index": { "type": "integer" }, "reason": { "type": "string" } },
                  "required": ["index", "reason"] }
            ],
            "additionalProperties": False
        }
    }
}