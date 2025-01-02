import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class RIVAL10_constants:
    # Example: "/home/ksas/Public/datasets/RIVAL10/{}/"
    _RIVAL10_DIR:str = None

    _LABEL_MAPPINGS = os.path.join(CURRENT_DIR, './data/label_mappings.json')
    _WNID_TO_CLASS =  os.path.join(CURRENT_DIR, './data/wnid_to_class.json')

    _ALL_CLASSNAMES = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]

    _ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
                'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 
                'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']
    
    _ZERO_SHOT_ATTRS = [
        'an animal with long-snout', 
        'an animal with  wings', 
        'a vehicle with wheels', 
        'has text written on it', 
        'an animal with  horns', 
        'an animal with floppy-ears', 
        'an animal with ears', 
        'an animal with colored-eyes', 
        'an object or an animal with a tail', 
        'an animal with mane', 
        'an animal with beak', 
        'an animal with hairy coat', 
        'an object with a metallic body', 
        'an object with rectangular shape', 
        'is damp, wet, or watery ', 
        'a long object', 
        'a tall object', 
        'has patterns on it']
    
    _KEY_FEATURES = {
        "car" : ["wheels", ],
        "plane" : ["wings"],
        "cat" : ["colored-eyes"],
        "bird" : ["wings"],
        "deer" : ["horns", "colored-eyes"],
        "dog" : ["floppy-ears"],
    }

    @staticmethod
    def set_rival10_dir(path:str):
        __class__._RIVAL10_DIR = os.path.join(path, "{}/")