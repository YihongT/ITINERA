def get_system_prompt(maxPoiNum, numMustSee, numCandidates):
    
    if numMustSee > maxPoiNum-2:
        maxPoiNum = numMustSee+2

    system_prompt = f"""
    Hello ChatGPT, please act as a travel master, skilled in curating exquisite travel experiences. Your mission is to create the perfect one-day itinerary from the provided list of 'potential points of interest (POI)'. Your specialty? Weaving vivid stories and evoking such a strong sense of wanderlust that just hearing your narrative feels like being there in person.

    ### Task Overview
    Your task is to return the perfect one-day itinerary based on the provided points of interest. Please return the itinerary directly in JSON format, without writing any code.
    """

    return system_prompt



def check_final_reverse_prompt(context, user_reqs):
    prompt = f"""
    Hello ChatGPT, here is a list or a text description of some locations or points. Please help me determine if most of the bars are located in the first half or the second half, and return your assessment in a JSON list format.

    ### Task Overview
    Your task is to
    1. First, based on the **user requirements**, determine if the user wants the starting point bar to be in the first half or the second half.
    2. If the user has no specific requirement for the starting point, use the provided **context** to assess whether most of the bars are in the first half or the second half.

    Please return the result directly as a JSON list based on the given context, without writing any code.

    ### Context
    {context}

    ### User Requirements
    {user_reqs}

    ### Output Example:
    - Your response should only be ["0"] or ["1"]
    - The response should not include any additional content.

    ### Output Specifications
    - Return a list with a single element containing an integer as a string, "0" indicating the first half, and "1" indicating the second half.
    - Return as a JSON list containing only one element.
    - The list may be empty; please return only a JSON list.
    - There should be no additional information in the output, and ensure the response can be parsed by json.loads.

    Now, based on the **context**, please provide your assessment following the **output specifications**.
    """
    
    return prompt


def get_hour_prompt(user_reqs):

    prompt = f"""
    Hello ChatGPT, please act as a top-level AI time-planning assistant. Your task is to determine the time required for a one-day itinerary based on the user's requirements. If the user requirements are empty, please default to returning ["4"].

    ### Task Overview
    Your task is to return the required time for a one-day itinerary based on the given user requirements. If the user specifies a time requirement for the itinerary in their request, return the requested time directly, but no more than 8 hours (if over 8 hours, return ["8"]). Please return the itinerary time directly based on user requirements, without writing any code.

    ### User Requirements
    {user_reqs}

    ### Input and Output Examples
    - **Example 1**:
    - **User Requirements**: ["Museum", "Authentic Cuisine", "Nightlife"]
    - **Output**: ["8"]
    
    - **Example 2**:
    - **User Requirements**: ["Historical Architecture Tour", "Cityscape Views"]
    - **Output**: ["6"]

    - **Example 3** (In this example, the user has specified **around five hours** for the itinerary):
    - **User Requirements**: ["Huangpu River", "Yuyuan Garden visit", "around five hours"]
    - **Output**: ["5"]

    ### Output Specifications
    - Return a list with a single element containing an integer, representing the required duration of the itinerary in hours. The range should be between 1 and 8.
    - Return as a JSON list containing only one element.
    - The list may be empty; please return only a JSON list.
    - There should be no additional information in the output, and ensure the response can be parsed by json.loads.

    Now, based on the **user requirements**, please return the required time for a one-day itinerary following the **output specifications**.
    """
    
    return prompt


def get_start_point_prompt(candidate_points, user_reqs, return_candidates, distance_string):

    candidate_strings = f""
    for i, candidate in enumerate(candidate_points):
        candidate_strings += f"Index {i}: {candidate}\n"

    prompt = f"""
    Hello ChatGPT, please act as a top-level AI travel planning assistant. Your task is to determine the best starting point index for a one-day itinerary based on the user requirements and provided candidate points. If the candidate points are empty, please default to returning ["0"].

    ### Task Overview
    Your task is to return the best starting point index for a one-day itinerary based on the given candidate points and user requirements. Please return the starting point index directly based on the user requirements and candidate points, without writing any code.

    ### Candidate Points
    {candidate_strings}

    ### User Requirements
    {user_reqs}

    ### Guidelines
    1. Ensure the selected point meets the user’s requirements: {user_reqs}
    2. The starting point should be close to its neighboring points.
    3. Prioritize locations like museums or art galleries, as they often require more time for exploration.
    4. Avoid starting with bars or clubs.

    ### Input and Output Examples
    - **Example 1**:
    - **Candidate Points**: ["Museum", "Park", "Bar"]
    - **Output**: ["0"]
    
    - **Example 2**:
    - **Candidate Points**: ["Shopping Center", "Art Gallery", "Historical Building"]
    - **Output**: ["1"]

    ### Output Specifications
    1. Return a list with a single element containing an integer, representing the index of the best starting point.
    2. Return as a JSON list containing only one element.
    3. There should be no additional information in the output, and ensure the response can be parsed by json.loads.
    4. Your response should be a JSON list of length 1 with an index from {return_candidates}.
        - Example: ["0"]

    Now, based on the **candidate points** and **user requirements**, please return the best starting point index for a one-day itinerary following the **output specifications**.
    Note: Ensure your response is a **list containing one index** from {return_candidates}, following the **output specifications**. The response should be a **JSON list of length 1** and should not contain any other content.
    """

    return prompt


def process_input_prompt(user_input):

    prompt = f"""
    Hello ChatGPT, please help me break down a user requirement description into multiple independent requirements. Each independent requirement should include both positive and negative aspects. Please directly return the result according to the following format based on the **user input**, without writing any code.

    ---

    ### Output Format:
    
    Return a list where each item is a dictionary representing an independent requirement, with the following key-value pairs:
    - **pos**: Positive requirement, representing what the user wants, excluding any negative requirements.
    - **neg**: Negative requirement, usually representing what the user does not want or wants to avoid; all negative aspects should be extracted into this field. For example, “not spicy” should extract “spicy,” “don’t want crowded” should extract “crowded,” and “dislikes noisy places” should extract “noisy.”
    - **mustsee**: Indicates whether this requirement represents a specific location name. If it does, this field is `true`; otherwise, it is `false`.
    - **type**: Indicates whether the requirement is for a “location” or the “itinerary,” with the values “location,” “starting point,” “ending point,” or “itinerary.”

    - Your return should be a list in the following format:
    [
        {{
            "pos": "positive requirement", (remaining requirements after excluding negative aspects)
            "neg": "negative requirement" (undesired, disliked, avoided, unwanted, unappealing aspects, any negatively expressed requirement),
            "mustsee": true (whether this is a mandatory location; all specific locations should be set to true),
            "type": "location" 
        }}, 
        ...
    ]
    - **Positive requirements** cannot be empty, and the positive requirement must not contain any negation; all negated aspects should be included in the "neg" field.
    - If there is no **negative requirement** for a particular location, set it to null.
    - Users may sometimes only describe what they do not want (negative requirement). In cases without a **positive requirement**, summarize a **positive requirement** based on the **negative requirement**. For example, if a user says "I don't want spicy food," the output should include: "pos" as "food," and "neg" as "spicy."
    - Independent requirements must have specific details or conditions to count as requirements; for instance, "recommend a route" does not qualify as an independent requirement.
    - The "mustsee" field must be `true` for specific location names, not for generic terms.
    - If a location is specifically a “starting point” or “ending point,” then the “type” field should be “starting point” or “ending point”; starting and ending points are required points, so "mustsee" should be set to true.
    - A location can only be a specific landmark or location to qualify as a “starting point” or “ending point”; you can only return at most one “starting point” and one “ending point.”
    - Do not include any additional content in the output.

    ### Output Examples:

    Example 1:
    Input: "I’d like to start by visiting Sinan Mansion, then find some fun activities nearby. Avoid crowded places."
    Output:
    [
        {{
            "pos": "Sinan Mansion",
            "neg": null,
            "mustsee": true,
            "type": "starting point"
        }}, 
        {{
            "pos": "fun places near Sinan Mansion", // if the "pos" field includes terms like "nearby," ensure it specifies the nearby location, e.g., “fun places near Sinan Mansion.”
            "neg": "crowded",
            "mustsee": false,
            "type": "location"
        }}
    ]

    Example 2:
    Input: "Stroll around Tian’ai Road, find a non-spicy local dish nearby, then see the night view at The Bund in the evening."
    Output:
    [
        {{
            "pos": "Tian’ai Road",
            "neg": null,
            "mustsee": true,
            "type": "starting point"
        }},
        {{
            "pos": "local cuisine restaurant near Tian’ai Road",
            "neg": "spicy",
            "mustsee": false,
            "type": "location"
        }},
        {{
            "pos": "The Bund",
            "neg": null,
            "mustsee": true,
            "type": "ending point"
        }}
    ]

    Example 3:
    Input: "Stroll around Haidian District"
    Output:
    [
        {{
            "pos": "Stroll around Haidian District",
            "neg": null,
            "mustsee": true,
            "type": "location"
        }}
    ]

    ---

    ### mustsee Field Assignment Examples
    "mustsee" is true for specific location names: “Hualian Mall,” “Old Mac Café,” “Wukang Mansion,” “Nanluoguxiang,” ...
    "mustsee" is false for generic location names: “mall,” “tea shop,” “bar,” “coffee,” ...
    
    ---

    ### Output Specifications
    - Return a list where each item is a dictionary containing the four key-value pairs "pos," "neg," "mustsee," and "type."
    - Return as a JSON list.
    - The list may be empty; if empty, please return only a JSON list.
    - There should be no additional information in the output, and ensure the response can be parsed by json.loads.

    ### User Input
    {user_input}

    ---

    ### Task Overview
    Your task is to analyze and break down the **user input** requirements into independent requirements and return them.
    1. First, split out different independent requirements and break each into positive and negative aspects.
    2. Positive requirements should only contain what the user wants, and negative aspects should be in the "neg" field.
    3. For each independent requirement, assign the "mustsee" field according to the **mustsee Field Assignment Examples** and analyze if the **positive requirement** represents a specific location name; if so, set "mustsee" to true; otherwise, set to false.
    4. Complete the other fields by following the **examples** and **output format**.

    #### Notes:
    - Avoid duplicate independent requirements and ensure each independent requirement corresponds to a different key point in the user input.
    - An “itinerary” requirement should be for the overall itinerary, such as including multiple locations, approximate time, etc., while others should be location requirements.
    - "itinerary" or "location" requirements are not mandatory to include; determine based on user input.
    - Ensure all negative requirements are extracted into the "neg" field.
    - The "pos" field must not contain any negations (like "not," "do not want," etc.), as all negatives should be in the "neg" field.
    - The "type" field can only be one of ["location", "itinerary", "starting point", "ending point"].

    Ensure that keywords accurately capture all key aspects of the requirements and keep descriptions concise and clear.

    All landmarks must be fully separated; for example, “Nanluoguxiang and Drum Tower” must be split into the two separate requirements “Nanluoguxiang” and “Drum Tower.”

    ---

    Now, based on the **user input**, refer to the **examples** and return the results according to the **output specifications** and **output format**.
    """

    return prompt


def get_dayplan_prompt(context_string, must_see_string, keyword_reqs, userReqList, maxPoiNum, numMustSee, numCandidates, itinerary_reqs, start_end, comments="", hours=None, mark_citywalk=False):
    
    if "fun" in keyword_reqs:
        keyword_reqs.remove("fun")

    if numMustSee > maxPoiNum-2:
        maxPoiNum = numMustSee+2

    if len(comments) > 0:
        comments = f"- Include comments in 'Overall Reason' in your response: {comments}"

    if hours is None:
        hours = 8

    if mark_citywalk == True:
        times = 2
    else:
        times = 1

    itinerary_pos_reqs, itinerary_neg_reqs = itinerary_reqs
    itinerary_reqs = f""
    if len(itinerary_pos_reqs) > 0:
        itinerary_reqs += f"- **Requirements for the itinerary**: {itinerary_pos_reqs}, "
    
    if len(itinerary_neg_reqs) > 0:
        itinerary_reqs += f"itinerary **should not include {itinerary_neg_reqs}**"
        
    start_poi, end_poi = start_end
    if start_poi is not None:
        start_poi = f'- **Starting Point**: {start_poi}'
    else:
        start_poi = f""
    if end_poi is not None:
        end_poi = f'- **Ending Point**: {end_poi}'
    else:
        end_poi = f""

    prompt = f"""
    Hello ChatGPT, I invite you to become a highly creative and knowledgeable travel guide, dedicated to designing the perfect itinerary for a one-day trip. Please thoughtfully create an itinerary in the form of an engaging and realistic travel story based on the provided list of “candidate points of interest.”

    ---

    ## Itinerary Design Rules

    Please carefully follow my instructions to create a memorable one-day itinerary for travelers.

    Design a one-day itinerary for travelers:
    - **Candidate Points Order**: {context_string}
    - **Must-See Points**: {must_see_string}
    - **Keyword Requirements**: {keyword_reqs}
    - **Original User Requirements**: {userReqList}
    {comments}
    {itinerary_reqs}
    {start_poi}
    {end_poi}

    ## Itinerary Constraints

    - **Itinerary Time**: Less than {hours} hours
    - **Point Selection**: Must follow the provided sequence order of points

    ---
        
    ## Output Format:
    {{
        "itinerary": "List of points separated by '->'",
        "Overall Reason": "Overall recommendation for the designed one-day itinerary",
        "pois": {{
            "n": "Description and recommendation for each point", ...
        }}
    }}
    Note:

    - "n" is the sequence number, which should be an integer. The sequence numbers in the output must be in ascending order, and they must match the numbers of selected points in the candidate list.
    - "itinerary" lists all visited points by name, separated by '->', e.g., "poi1->poi2->...", noting that it should only contain names without sequence numbers, and the order must exactly match the "pois" point order.
    
    ---

    ## Language Style Transformation:

    1. Refer to the **Style Example** below to rewrite descriptions in the “Candidate Points” list:
    - Writing Style: Use a **prose** style to expand and enrich the features or activities, adding sensory details and atmospheric descriptions. Employ expressive, emotional phrases to make descriptions vivid and engaging.
    - Vivid Imagery: Use descriptive adjectives and phrases to provide a lively visual experience for readers.
    - Contrast and Comparison: Use contrast techniques, like comparing old with modern, busy scenes with quiet surroundings, to highlight each point’s uniqueness.
    - Specific Location and Emotional Connection: Clearly mention specific locations or environments, combining with historical or emotional background to enhance authenticity and depth.
    - Use varied sentence structures and styles for different locations to ensure diversity of expression.

    ### Style Example:
    -  Original Description: Jing'an Temple, Jing'an District, a splendid and grand temple in the heart of the bustling city
       Transformed: Amid the bustling Jing’an District lies the ancient Jing’an Temple, a place that feels remarkably different. Its golden splendor and majestic aura evoke a sense of tranquility and dignity with a unique charm.
    -  Original Description: Montreal Garden, Pudong New District, a beautiful spot for photography; in autumn, the avenue of sycamore trees glows golden, while in summer, the lotus blooms in the pond
       Transformed: Montreal Garden in Pudong New District is a photographer’s paradise. In autumn, the sycamore-lined avenue glows with a golden hue, while in summer, the lotus flowers bloom magnificently, captivating every passerby.
    -  Original Description: By Suzhou River, Huangpu District
       Transformed: Alongside the Suzhou River in Huangpu District, the gentle river murmurs tales of the city’s past, drawing in people to pause and reflect.

    2. For each transformed description, vary expressions and sentence styles to ensure diversity in tone.

    ---

    -- Thoughts Before Action --
    1. Take a deep breath and work through each step methodically.
    2. THINK HARD AND THOROUGHLY; **do not skip, simplify, or shorten**. THIS IS VERY IMPORTANT TO ME.
    3. Ensure that travelers feel fully immersed, making the itinerary feel custom-made for them.
    4. **Only select** points from the **Candidate Points** list and arrange them in **ascending sequence order**.
    5. Do not include bars in the itinerary unless specifically requested in **Keyword Requirements**.
    6. Limit cafes and bars to no more than two, adhering to the **Candidate Points order**. Bars should be at the end, while cafes should not be the final stop.
    7. If adding a bar at the end disrupts the ascending order of points, **do not include that bar**.
    8. Exclude amusement parks (like Disneyland) unless specifically requested in the **Original User Requirements**.
    9. Limit the itinerary to **one dining location** at most.
    10. Strictly meet each **Keyword Requirement**, ensuring that the itinerary exactly matches the number of points specified.

    ---
    
    Please consider each point’s opening hours and visit duration to ensure all selected points can realistically be visited within the day.

    ## Itinerary Generation Steps
    1. Based on the **Candidate Points** list, select appropriate points in ascending order to add to the itinerary, aiming for around 6-10 points, and do not add all points from the **Candidate Points**.
    2. All points in the itinerary must follow the ascending order of the **Candidate Points** list.
    3. Exclude any cafes or bars if their addition disrupts the ascending order of points.
    4. Ensure that every **Keyword Requirement: {keyword_reqs}** is met by at least one point in the **Candidate Points Order**.
    5. **Original User Requirements: {userReqList}** should also be carefully considered and met by points in the **Candidate Points Order**.
    6. Generate a JSON file containing all selected points.
    7. The **Overall Reason** should be concise, primarily summarizing the itinerary theme and features, following the **Language Style Transformation**, and should not exceed 50 words.
    8. Describe each point individually without merging into a single journey narrative; each point should be introduced independently, drawing on the **Language Style Transformation** for varied styles and ensuring diversity in expression.

    ---
    
    Now, please design a selected itinerary with 6-10 points for travelers following the **Itinerary Generation Steps** and **Itinerary Constraints**, then convert the original descriptions using **Language Style Transformation** to describe the itinerary.
    Note:
    - The recommendation for the first point should begin with “Our itinerary starts at...”, or similar phrasing. Each point should transition smoothly to the next with phrases like “Leaving XX, we continue to XXX,” or “Walking along XX road, we reach XXX.” The recommendation should blend with the itinerary description.
    - The **Overall Reason** and each point’s **Point Recommendation** should be approximately 30 words, with a maximum of 40 words. The recommendation should be concise and engaging, presenting an objective description while omitting all first-person subjective language.
    - Points must be **selected**, with 6 to 10 points in total. **Point names must come from the **Candidate Points** list**.
    - **Ensure all selected points come from the **Candidate Points Order** and are arranged in ascending order**, and **do not select all candidate points**.
    """
    
    return prompt
