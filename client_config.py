
client_names = [
    'Ups',  
    'Sti', 
    'Aircargo', 
    'Bode', 
    'Login', 
    'Nagel',
    'Ktn',
    'Kipfer',
    'Lindt',
    'Lgi',
    'Bork'
]

client_configs = {
    # 'Nagel': {
    #     'truck_range': range(21, 46),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Nagel Locations', 'Food', 'Meat', 'Dairy', 'Baker', 'Other Logistics', 
    #         'Retailers', 'Supermarket', 'Other Customers', 'Highway Rest place', 
    #         'Gas Stations', 'Unknown Locations', 'Maintaince/Workshop'
    #     ]
    # },
    'Bode': {
        'truck_range': range(21, 40),
        'sensor_orientations': {},
        'location_categories': [
            'Bode Locations', 'Lidl', 'Terminals', 'Other Customers', 'Other Logistics', 
            'Food', 'Transit Stations', 'Maintaince/Workshop', 'Packaging', 
            'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    # 'Quehenberger': {
    #     'truck_range': range(1, 3),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Quehenberger Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
    #     ]
    # },
    # 'Greilmeier': {
    #     'truck_range': range(3, 5),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Greilmeier Locations', 'Food', 'Meat', 'Dairy', 'Baker', 'Other Logistics', 
    #         'Retailers', 'Supermarket', 'Other Customers', 'Highway Rest place', 
    #         'Gas Stations', 'Unknown Locations', 'Maintaince/Workshop'
    #     ]
    # },
    'Kipfer': {
        'truck_range': range(3, 5),
        'sensor_orientations': {},
        'location_categories': [
            'Kipfer Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations', 'Pharmaceutical'
        ]
    },
    'Login': {
        'truck_range': range(3, 6),
        'sensor_orientations': {},
        'location_categories': [
            'Login Locations', 'BAT Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations', 'Tobacco'
        ]
    },
    # 'Jjx': {
    #     'truck_range': range(1, 3),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Jjx Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
    #     ]
    # },
    # 'Konvoi': {
    #     'truck_range': range(25, 26),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Highway Rest place', 'Gas Stations', 'Unknown Locations'
    #     ]
    # },
    'Aircargo': {
        'truck_range': range(1, 8),
        'sensor_orientations': {},
        'location_categories': [
            'Aircargo Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    'Ups': {
        'truck_range': range(1, 3),
        'sensor_orientations': {},
        'location_categories': [
            'Ups Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations', 'Pharmaceutical'
        ]
    },
    'Sti': {
        'truck_range': range(1, 3),
        'sensor_orientations': {},
        'location_categories': [
            'Sti Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    'Lindt': {
        'truck_range': range(1, 9),
        'sensor_orientations': {},
        'location_categories': [
            'Lindt Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },

    'Lgi': {
        'truck_range': range(1, 4),
        'sensor_orientations': {},
        'location_categories': [
            'Lgi Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    # 'Helveticor': {
    #     'truck_range': range(1, 3),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Helveticor Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
    #     ]
    # },
    # 'Emde': {
    #     'truck_range': range(1, 3),
    #     'sensor_orientations': {},
    #     'location_categories': [
    #         'Emde Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
    #     ]
    # },
    'Ktn': {
        'truck_range': range(1, 5),
        'sensor_orientations': {},
        'location_categories': [
            'Ktn Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    'Lgi': {
        'truck_range': range(1, 3),
        'sensor_orientations': {},
        'location_categories': [
            'Lgi Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    'Lindt': {
        'truck_range': range(1, 9),
        'sensor_orientations': {},
        'location_categories': [
            'Lindt Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    },
    'Bork': {
        'truck_range': range(1, 9),
        'sensor_orientations': {},
        'location_categories': [
            'Bork Locations', 'Highway Rest place', 'Gas Stations', 'Unknown Locations'
        ]
    }
}

def get_client_trucks(client_name):
    """Get list of truck names for a specific client"""
    if client_name not in client_configs:
        return []
    return [f"{client_name}_{i}" for i in client_configs[client_name]['truck_range']]

def get_all_trucks():
    """Get list of all trucks across all clients"""
    all_trucks = []
    for client_name in client_names:
        all_trucks.extend(get_client_trucks(client_name))
    return all_trucks

def validate_client_name(client_name):
    """Validate if client name exists in configuration"""
    return client_name in client_configs

# Configuration validation (runs when imported)
def _validate_config():
    """Internal function to validate configuration consistency"""
    errors = []
    
    # Check if all client names have configurations
    for client_name in client_names:
        if client_name not in client_configs:
            errors.append(f"Client '{client_name}' in client_names but not in client_configs")
    
    # Check if all configurations have corresponding names
    for client_name in client_configs:
        if client_name not in client_names:
            errors.append(f"Client '{client_name}' in client_configs but not in client_names")
    
    # Check required keys in each config
    required_keys = ['truck_range', 'sensor_orientations', 'location_categories']
    for client_name, config in client_configs.items():
        for key in required_keys:
            if key not in config:
                errors.append(f"Client '{client_name}' missing required key '{key}'")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    return len(errors) == 0

# Run validation when module is imported
_validate_config()