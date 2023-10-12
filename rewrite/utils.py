coco_caption_list = [
['A herd of goats walking down a road way.', 'Three lambs stand next to each other and look different directions. ', 'The animals standing in the clearing are 3 varieties of sheep.', 'Three small sheep are standing on a road.', 'Some animals are standing on a dirt path'],
['A boy is preparing to toss a frisbie while another boy is sitting in the background in a park.', 'Several people are out in the woods on a path playing a game.', 'A man in a park playing a throwing game.', 'A group of people that are hanging out together.', 'A boy gets ready to throw a frisbee'],
['A pizza sitting on top of a metal pan.', 'The large pepperoni pizza is covered with chives.', 'A pizza that is sitting on a tray.', 'A large pizza with toppings sitting on a tray.', 'a pizza with fresh basil tomato sauce and cheese baked'],
['A woman sits on top of a motorcycle in a parade.', 'Woman wearing starts on helmet and shorts rides motorcycle', 'A woman wearing attire that matches her motorcycle is driving on. ', 'A person that is on top of a motorcycle.', 'Woman on a motorcycle rides in a parade'],
['the people are sampling wine at a wine tasting.', 'Group of people tasting wine next to some barrels.', 'People are gathered around a man tasting wine.', 'A man pouring wine from casks for patrons', 'People gather around a table while sampling wine.'],
['A herd of sheep walking down a street in front of a bus.', 'There are three animals walking down the road.', 'a van is stuck behind a few traveling goats', 'a van that has some kind of animal out front of it', 'A herd of animals walking down the road behind a truck.'],
['A sandwich with meat and cheese sits on a plate with a small salad.', 'A sandwich with cheese and a bowl with a salad.', 'Two plates with sandwiches on them next to a bowl of vegetables.', 'A long sandwich and a salad is on a plate.', 'a sandwich and a bowl of vegetables on a plate '],
['A NASA airplane carrying a space shuttle on its back.', 'A large plan with a smaller plan on top of it. ', 'A NASA airplane carrying the old Space Shuttle', 'A NASA airplane glides through the sky while carrying a shuttle.', 'This jet is carrying a space shuttle on it'],
['A one way sign under a blue street sign. ', 'a view from below of a one way sign ', 'A street sign stating that the road is one way beneath a blue sky. ', 'A "One Way" street sign pointing to the right. ', 'A one way road sign mounted above a street sign.'],
['A bowl of food containing broccoli and tomatoes. ', 'A large salad is displayed in a silver metal bowl.', 'A bowl of food with tomatoes, sliced apples, and other greens', 'A silver bowl filled with various produce discards.', 'The salad in the bowl contains many fresh fruits and vegetables.  '],
['a cake made to look like it has candy decorations on it ', 'A photograph of a highly decorated cake on a table.', 'A cake decorated with lollipops and a piece of pie.', 'A piece of cake with  lolypops, pie and caterpillar designs.', 'A layered cake with sweet treats and a caterpillar as decorations.'],
['A young man riding a skateboard on a cement walkway.', 'a guy riding a skateboard by a car ', 'A young man on a skateboard near a car', 'an image of a boy on a skateboard doing tricks', 'A young man is riding on his skateboard.'],
['A small brown dog sitting on display behind a window.', 'A small fuzzy dog stares longingly out a window.', 'The dog is  brown shaggy with a red collar. ', 'A dog sits alone and stares out of a window.', 'A furry and cute dog sitting in a window looking outside.'],
['A herd of sheep standing on a lush green hillside.', 'Several animals standing on the side of a hill.', 'A number of sheep eat on a steep grassy hill.', 'a couple of sheep are standing in some grass', 'The side of a small hill of grass with several sheep grazing in the grass and houses in the background on the upper hill.'],
['The tennis player on the blue court has his racquet raised.', 'A man swinging a tennis racket at a pro tennis match.', 'A tennis player wearing a NIKE shirt swings his racket', 'Man posing in front of the camera holding up a tennis racket.', 'A man wearing a white shirt playing tennis.'],
['A surfer riding a wave in a tempestuous ocean', 'Man in body suit surfing on a large wave.', 'A surfer is sideways on a wave of water on a surfboard.', 'The surfer is riding sideways along a wave. ', 'a surfer wearing a wet suit is surfing on a white board'],
['A woman with a sweater over her eyes.', 'A girl who is blindfolded is trying to bite a donut on a string held up by another person.', 'a woman covering her eyes playing a game', "Man holding donut on stick and string above woman's covered head.", 'a person is trying to bite a doughnut blindfolded'],
['A man is petting the face if a tame horse', "A man petting a horse's face while a woman stands behind next to another horse.", 'Black and white image of a woman and a man petting a horse.', 'A man is petting a horse and a woman is standing next to him.', 'A black and white photo of a man petting a horse'],
]



chatgpt_source_caption_list = [
    'white and red cheerful combination in the bedroom for girl',
    'vintage photograph of a young boy feeding pigeons .',
    'businessman with smartphone sitting on ledge by the sea',
    'a tourist taking a photograph of river looking west towards suspension bridge and office',
    'glass of foods and food product on a sunny day',
    'turtles and large fish in the pond',
    'the frescoes inside the dome',
    'fight over a loose ball',
    'love this winter picture by person .',
    'one of several paths through the woods .',
    'ripe strawberries falling through the water .',
    'a city reflected on a red sunglasses .',
    'man driving a car through the mountains',
    'maritime museum from the historical dock .',
    'tree hollow and green leaves of a tree top in summer',
    'musician of musical group performs on stage on the first day of festival',
]

chatgpt_target_caption_list = [
    'A bright and lively white-and-red color scheme in a girl\'s bedroom, creating a cheerful ambiance.',
    'A charming vintage photograph capturing a young boy feeding a flock of pigeons in a bustling city square.',
    'Serene coastal view as a businessman sits on a ledge by the sea, using his smartphone.',
    'Tourist snaps photo of suspension bridge and office building across the river.',
    'An assortment of food items and products displayed in a glass container, illuminated by bright sunshine.',
    'A tranquil pond where large fish and turtles coexist peacefully, creating a harmonious natural habitat.',
    'The elaborate and intricate paintings or artworks adorning the inner surface of the dome, typically found in religious buildings.',
    'Intense competition as players struggle to gain control of a loose ball during the game.',
    'Mesmerizing winter landscape by person: serene snowy scenery with gentle snowflakes, skillfully framed with perfect contrast and depth.',
    'A narrow forest path, one among many weaving through the lush trees, underbrush, and dappled sunlight.',
    'Juicy ripe strawberries plummeting through a stream of water, splashing and creating ripples in the liquid.',
    'The cityscape reflected on a pair of red sunglasses, creating a distorted but fascinating view of the urban environment.',
    'A man confidently navigating a winding mountain road with breathtaking views.',
    'A museum dedicated to seafaring history, located on a historic dock where visitors can view a collection of artifacts and vessels.',
    'Amidst lush green leaves on the top of a tree, a hollow creates a natural shelter, typical of summer foliage.',
    'On the opening day of the festival, a musician from a musical group performs energetically on stage to a lively crowd.'
]


bard_source_caption_list = [
    'man driving a car through the mountains',
    'a bicycle hanging above the entrance to a store',
    'government agency released underwater footage of the unique movements of starfish',
    'unique red chair among other white chairs at the stadium',
    'person looks comfortable as he connects with a free - kick during the session and is in line to return against hull on saturday',
    'animal in front of a white background',
    'a mother and daughter lying on a lawn',
    'sign is seen outside the home',
    'portrait of person against an abstract background stock photo',
    'state flag waving on an isolated white background .',
    'actor wears a gorgeous blush pink coloured gown at festival .',
    'person answering the phones again at the office .',
    'little boy sitting on the grass with drone and remote controller',
    'golfer competes during day held',
    'golden fish in a bowl',
    'businessman with smartphone sitting on ledge by the sea',
]
bard_target_caption_list = [
    'A man drives his car through the mountains, the road winding its way through the towering peaks.',
    'A bicycle suspended from a chain above the entrance to a bike shop advertises their services.',
    'New footage released by government agency shows starfish swimming in unique ways, shedding light on their underwater behavior.',
    'Red chair stands out among white seats at stadium, a beacon of color.',
    'The player strikes a free-kick with confidence during training, looking sharp and ready for his return to the team against Hull on Saturday.',
    'An animal stands out against the stark white background, its feathers on full display.',
    'A mother and daughter embrace in the grass, enjoying the warmth of the sun and the sound of birdsong.',
    'A for sale sign stands in the front yard, a reminder that change is always on the horizon.',
    'person\'s face stands out against a swirling, abstract background, their expression one of mystery and intrigue.',
    'The state flag waves proudly against a stark white background. The colors of the flag are vibrant and the design is intricate, representing the history and culture of the state',
    'Actor stuns in blush pink gown at festival, her beauty a match for the flowers in bloom.',
    'A friendly voice answers the phone, eager to help the caller with their needs.',
    'A young boy\'s eyes light up as he launches his drone into the sky, his imagination taking flight with it.',
    'Golfer focuses on the ball as she competes in a tournament on a beautiful day.',
    'A golden fish swims lazily in a bowl, its scales shimmering in the sunlight. The fish is a beautiful shade of orange, with black spots on its fins and tail.',
    'A businessman relaxes on a seaside ledge, checking his phone and enjoying the view.'
]


human_source_caption_list = [
    'Honey buttermilk biscuits on a cooling rack being drizzled with honey',
    'happy corgi time',
    '<PERSON> dog looking at dirt from the ground',
    'navy vintage pants - lime green bag - ivory Maison Simons t-shirt - Zara clogs',
    'Ooak Barbie City Shine',
    'Real Wedding on a NYC Rooftop',
    'the proud of my beloved italian bracco after leg amputation due to a tumor.',
    'Pineapple Wearing Headphones Art Print by Philip Haynes',
    'Ominous thunderclouds behind the Capitol Building',
    'Steampunk woman with gun',
    'a new watch with some old friends',
    'Particularly important to Africa is the East African Highland Banana (EAHB), a staple food for 80 million people. Uganda alone has about 120 varieties of this type of banana.',
    'Electric Blue Guitar There Goes My Hero, Rock The Vote, <PERSON>, <PERSON>, Music Photo, Red Eyes, Photo Quotes, Electric Blue, Music Lyrics',
    'Advanced Bicycle Skills Video - Valuable Video for Safe Cycl',
    'grilled turkey pesto sandwich',
    'Actress <PERSON> during the launch of international fashion brand Forever 21 store at a mall in Mumbai on Saturday, October 12th, 2013.'
]

human_target_caption_list = [
    'A warm stack of freshly baked honey buttermilk biscuits, sit on a cooling rack as they are drizzled with golden honey',
    'Delighted corgi stands in the hallway, looking at its owner',
    '<Person>\'s dog, lying on the ground, looks at the dirt',
    'A young beautiful lady wearing navy vintage pants and ivory Maison Simons t-shirt, is holding a lime green bag.',
    'A custom-made Barbie doll with a city-inspired look shines brightly',
    'a couple is kissing each other during their rooftop wedding in NYC',
    'my italian bracco lied down proudly under the sunshile, despite of leg amputation due to a tumor.',
    'An art from Philip Haynes depicts a pineapple that wears headphones',
    'Thunderclouds loom over the Capitol Building, casting a dark shadow',
    'A fierce and stylish steampunk woman holds a toy revolver in her hands',
    'The watch sits besides a cartoon picture, evoking memories of cherished times shared with long-time friends',
    'An African man holds a bunch of bananas, which is particularly important to Africa',
    '<PERSON> is playing an electric blue guitar, eyes bloodshot from the stage lights',
    'A Cyclist is demonstrating advanced bicycle skills in a video that will help people stay safe.',
    'A grilled turkey pesto sandwich with melted cheese and fresh arugula is served on a plate.',
    'The young beautiful actress attended the launch of fashion brand Forever 21 at a mall.'
]

