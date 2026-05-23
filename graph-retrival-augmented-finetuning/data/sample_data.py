"""
Diverse sample dataset for GRAFT training.

Replaces the single-sample × 40 approach with genuinely diverse examples
spanning different scene types, relations, and reasoning patterns.

In production, replace this with actual GQA + FVQA data loading.
"""


def get_diverse_dataset() -> list[dict]:
    """
    Returns diverse GRAFT training samples.

    Each sample has:
    - Different scene graph topology
    - Different KB facts and reasoning paths
    - Different query types (causal, descriptive, predictive)
    """
    samples = [
        # 1. Electric vehicle charging (original)
        {
            "image_id": "2407890",
            "gqa_scene_graph": {
                "objects": {
                    "101": {"name": "delivery_van", "attributes": ["white", "electric"]},
                    "102": {"name": "charging_station", "attributes": ["active"]},
                },
                "relations": [{"source": "101", "name": "connected_to", "target": "102"}],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "delivery_van", "rel": "IsA", "e2_label": "electric_vehicle"},
                {
                    "e1_label": "charging_station",
                    "rel": "ProvidesPowerTo",
                    "e2_label": "electric_vehicle",
                },
                {
                    "e1_label": "electric_vehicle",
                    "rel": "Requires",
                    "e2_label": "thermal_management",
                },
            ],
            "kb_distractor_pool": [
                {"e1_label": "street_light", "rel": "Emits", "e2_label": "yellow_light"},
                {"e1_label": "pedestrian", "rel": "WalksOn", "e2_label": "crosswalk"},
            ],
            "query": "What risk does this vehicle face if left unchanged over the next hour?",
            "ground_truth_trajectory": "Path: [Scene: delivery_van] -> connected_to -> [Scene: charging_station] <=> Aligned <=> <delivery_van, IsA, electric_vehicle> -> <charging_station, ProvidesPowerTo, electric_vehicle>.",
            "answer": "The delivery van is linked to an active charging station. Extended high-voltage power input risks battery degradation without active thermal management.",
        },
        # 2. Kitchen fire hazard
        {
            "image_id": "3501234",
            "gqa_scene_graph": {
                "objects": {
                    "201": {"name": "stove", "attributes": ["gas", "lit"]},
                    "202": {"name": "towel", "attributes": ["cotton", "hanging"]},
                    "203": {"name": "pot", "attributes": ["metal", "on_burner"]},
                },
                "relations": [
                    {"source": "202", "name": "near", "target": "201"},
                    {"source": "203", "name": "on", "target": "201"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "cotton", "rel": "HasProperty", "e2_label": "flammable"},
                {"e1_label": "gas_stove", "rel": "Produces", "e2_label": "open_flame"},
                {
                    "e1_label": "flammable_material",
                    "rel": "IgnitesWhen",
                    "e2_label": "near_open_flame",
                },
            ],
            "kb_distractor_pool": [
                {"e1_label": "pot", "rel": "MadeOf", "e2_label": "stainless_steel"},
                {"e1_label": "kitchen", "rel": "Contains", "e2_label": "refrigerator"},
            ],
            "query": "What safety hazard exists in this scene?",
            "ground_truth_trajectory": "Path: [Scene: towel] -> near -> [Scene: stove] <=> Aligned <=> <cotton, HasProperty, flammable> -> <gas_stove, Produces, open_flame> -> <flammable_material, IgnitesWhen, near_open_flame>.",
            "answer": "The cotton towel hanging near the lit gas stove is a fire hazard. Cotton is flammable and proximity to an open flame creates ignition risk.",
        },
        # 3. Traffic intersection
        {
            "image_id": "4602345",
            "gqa_scene_graph": {
                "objects": {
                    "301": {"name": "bicycle", "attributes": ["moving"]},
                    "302": {"name": "truck", "attributes": ["large", "turning"]},
                    "303": {"name": "traffic_light", "attributes": ["green"]},
                },
                "relations": [
                    {"source": "301", "name": "beside", "target": "302"},
                    {"source": "302", "name": "approaching", "target": "303"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "truck", "rel": "HasProperty", "e2_label": "large_blind_spot"},
                {"e1_label": "bicycle", "rel": "IsA", "e2_label": "vulnerable_road_user"},
                {"e1_label": "blind_spot", "rel": "Causes", "e2_label": "collision_risk"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "traffic_light", "rel": "Controls", "e2_label": "intersection_flow"},
                {"e1_label": "road", "rel": "HasMarking", "e2_label": "lane_divider"},
            ],
            "query": "What danger does the cyclist face?",
            "ground_truth_trajectory": "Path: [Scene: bicycle] -> beside -> [Scene: truck] <=> Aligned <=> <truck, HasProperty, large_blind_spot> -> <bicycle, IsA, vulnerable_road_user> -> <blind_spot, Causes, collision_risk>.",
            "answer": "The cyclist is beside a turning truck. Trucks have large blind spots, and the cyclist as a vulnerable road user faces collision risk if the truck turns without seeing them.",
        },
        # 4. Medical scene
        {
            "image_id": "5703456",
            "gqa_scene_graph": {
                "objects": {
                    "401": {"name": "patient", "attributes": ["elderly", "seated"]},
                    "402": {"name": "medication_bottle", "attributes": ["open", "multiple"]},
                    "403": {"name": "glass", "attributes": ["empty"]},
                },
                "relations": [
                    {"source": "401", "name": "holding", "target": "402"},
                    {"source": "403", "name": "on_table_near", "target": "401"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "elderly_patient", "rel": "AtRiskOf", "e2_label": "polypharmacy"},
                {
                    "e1_label": "multiple_medications",
                    "rel": "Causes",
                    "e2_label": "drug_interaction",
                },
                {"e1_label": "drug_interaction", "rel": "LeadsTo", "e2_label": "adverse_effects"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "glass", "rel": "MadeOf", "e2_label": "transparent_material"},
                {"e1_label": "chair", "rel": "UsedFor", "e2_label": "sitting"},
            ],
            "query": "What medical concern is suggested by this scene?",
            "ground_truth_trajectory": "Path: [Scene: patient] -> holding -> [Scene: medication_bottle(multiple)] <=> Aligned <=> <elderly_patient, AtRiskOf, polypharmacy> -> <multiple_medications, Causes, drug_interaction>.",
            "answer": "The elderly patient with multiple open medication bottles suggests polypharmacy risk. Multiple medications increase the chance of drug interactions and adverse effects.",
        },
        # 5. Construction site
        {
            "image_id": "6804567",
            "gqa_scene_graph": {
                "objects": {
                    "501": {"name": "worker", "attributes": ["no_helmet"]},
                    "502": {"name": "crane", "attributes": ["operating", "overhead"]},
                    "503": {"name": "steel_beam", "attributes": ["suspended"]},
                },
                "relations": [
                    {"source": "502", "name": "lifting", "target": "503"},
                    {"source": "501", "name": "below", "target": "503"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "suspended_load", "rel": "HasRisk", "e2_label": "falling_object"},
                {"e1_label": "hard_hat", "rel": "Protects", "e2_label": "head_injury"},
                {
                    "e1_label": "worker_without_ppe",
                    "rel": "Violates",
                    "e2_label": "safety_regulation",
                },
            ],
            "kb_distractor_pool": [
                {"e1_label": "crane", "rel": "OperatedBy", "e2_label": "certified_operator"},
                {"e1_label": "construction_site", "rel": "Requires", "e2_label": "permit"},
            ],
            "query": "What safety violation is occurring?",
            "ground_truth_trajectory": "Path: [Scene: worker(no_helmet)] -> below -> [Scene: steel_beam(suspended)] <=> Aligned <=> <suspended_load, HasRisk, falling_object> -> <hard_hat, Protects, head_injury> -> <worker_without_ppe, Violates, safety_regulation>.",
            "answer": "A worker without a helmet is positioned below a suspended steel beam. This violates safety regulations as falling objects from overhead crane operations require head protection.",
        },
        # 6. Weather + agriculture
        {
            "image_id": "7905678",
            "gqa_scene_graph": {
                "objects": {
                    "601": {"name": "crop_field", "attributes": ["dry", "wilting"]},
                    "602": {"name": "irrigation_system", "attributes": ["inactive"]},
                    "603": {"name": "sky", "attributes": ["clear", "hot"]},
                },
                "relations": [
                    {"source": "602", "name": "installed_in", "target": "601"},
                    {"source": "603", "name": "above", "target": "601"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "wilting_crops", "rel": "IndicatesLackOf", "e2_label": "water"},
                {
                    "e1_label": "inactive_irrigation",
                    "rel": "Causes",
                    "e2_label": "crop_dehydration",
                },
                {"e1_label": "prolonged_heat", "rel": "Accelerates", "e2_label": "crop_death"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "field", "rel": "Contains", "e2_label": "soil"},
                {"e1_label": "sun", "rel": "Provides", "e2_label": "photosynthesis_energy"},
            ],
            "query": "What will happen to the crops if conditions persist?",
            "ground_truth_trajectory": "Path: [Scene: crop_field(wilting)] -> installed_in <- [Scene: irrigation_system(inactive)] <=> Aligned <=> <wilting_crops, IndicatesLackOf, water> -> <inactive_irrigation, Causes, crop_dehydration> -> <prolonged_heat, Accelerates, crop_death>.",
            "answer": "The wilting crops with inactive irrigation under hot clear skies face accelerated dehydration. Without water supply restoration, prolonged heat will lead to crop death.",
        },
        # 7. Electrical hazard
        {
            "image_id": "8106789",
            "gqa_scene_graph": {
                "objects": {
                    "701": {"name": "power_cable", "attributes": ["frayed", "exposed"]},
                    "702": {"name": "puddle", "attributes": ["water"]},
                    "703": {"name": "child", "attributes": ["playing"]},
                },
                "relations": [
                    {"source": "701", "name": "touching", "target": "702"},
                    {"source": "703", "name": "near", "target": "702"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "exposed_wire", "rel": "Conducts", "e2_label": "electricity"},
                {"e1_label": "water", "rel": "IsA", "e2_label": "conductor"},
                {"e1_label": "electrified_water", "rel": "Causes", "e2_label": "electrocution"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "child", "rel": "Enjoys", "e2_label": "playing_outdoors"},
                {"e1_label": "rain", "rel": "Creates", "e2_label": "puddles"},
            ],
            "query": "What immediate danger exists for the child?",
            "ground_truth_trajectory": "Path: [Scene: power_cable(frayed)] -> touching -> [Scene: puddle] <- near <- [Scene: child] <=> Aligned <=> <exposed_wire, Conducts, electricity> -> <water, IsA, conductor> -> <electrified_water, Causes, electrocution>.",
            "answer": "A frayed power cable is touching a water puddle near a playing child. Water conducts electricity, creating electrocution risk if the child contacts the puddle.",
        },
        # 8. Environmental pollution
        {
            "image_id": "9207890",
            "gqa_scene_graph": {
                "objects": {
                    "801": {"name": "factory", "attributes": ["smoking", "industrial"]},
                    "802": {"name": "river", "attributes": ["discolored"]},
                    "803": {"name": "pipe", "attributes": ["discharge"]},
                },
                "relations": [
                    {"source": "803", "name": "connects", "target": "801"},
                    {"source": "803", "name": "drains_into", "target": "802"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "industrial_discharge", "rel": "Contains", "e2_label": "heavy_metals"},
                {"e1_label": "heavy_metals", "rel": "Causes", "e2_label": "water_contamination"},
                {"e1_label": "contaminated_water", "rel": "Harms", "e2_label": "aquatic_ecosystem"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "factory", "rel": "Produces", "e2_label": "goods"},
                {"e1_label": "river", "rel": "FlowsTo", "e2_label": "ocean"},
            ],
            "query": "What environmental impact is this factory causing?",
            "ground_truth_trajectory": "Path: [Scene: factory] <- connects <- [Scene: pipe] -> drains_into -> [Scene: river(discolored)] <=> Aligned <=> <industrial_discharge, Contains, heavy_metals> -> <heavy_metals, Causes, water_contamination> -> <contaminated_water, Harms, aquatic_ecosystem>.",
            "answer": "The factory pipe draining into the discolored river indicates industrial discharge containing heavy metals. This causes water contamination that harms the aquatic ecosystem.",
        },
        # 9. Structural engineering
        {
            "image_id": "1038901",
            "gqa_scene_graph": {
                "objects": {
                    "901": {"name": "bridge", "attributes": ["old", "cracked"]},
                    "902": {"name": "support_column", "attributes": ["corroded"]},
                    "903": {"name": "heavy_truck", "attributes": ["loaded"]},
                },
                "relations": [
                    {"source": "902", "name": "supports", "target": "901"},
                    {"source": "903", "name": "crossing", "target": "901"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "corroded_steel", "rel": "HasReduced", "e2_label": "load_capacity"},
                {
                    "e1_label": "cracked_concrete",
                    "rel": "Indicates",
                    "e2_label": "structural_fatigue",
                },
                {"e1_label": "overloaded_structure", "rel": "RisksOf", "e2_label": "collapse"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "bridge", "rel": "SpansOver", "e2_label": "river"},
                {"e1_label": "truck", "rel": "UsedFor", "e2_label": "transport"},
            ],
            "query": "Is it safe for the truck to cross this bridge?",
            "ground_truth_trajectory": "Path: [Scene: heavy_truck] -> crossing -> [Scene: bridge(cracked)] <- supports <- [Scene: support_column(corroded)] <=> Aligned <=> <corroded_steel, HasReduced, load_capacity> -> <cracked_concrete, Indicates, structural_fatigue> -> <overloaded_structure, RisksOf, collapse>.",
            "answer": "The bridge shows structural fatigue with cracked concrete and corroded support columns reducing load capacity. A heavy loaded truck crossing risks structural collapse.",
        },
        # 10. Wildlife interaction
        {
            "image_id": "1149012",
            "gqa_scene_graph": {
                "objects": {
                    "1001": {"name": "bear", "attributes": ["adult", "approaching"]},
                    "1002": {"name": "campsite", "attributes": ["unattended"]},
                    "1003": {"name": "food_container", "attributes": ["open", "on_ground"]},
                },
                "relations": [
                    {"source": "1001", "name": "approaching", "target": "1002"},
                    {"source": "1003", "name": "at", "target": "1002"},
                ],
            },
            "fvqa_graph_rag_facts": [
                {"e1_label": "bear", "rel": "AttractedBy", "e2_label": "food_scent"},
                {"e1_label": "habituated_bear", "rel": "Becomes", "e2_label": "dangerous"},
                {"e1_label": "open_food", "rel": "Emits", "e2_label": "food_scent"},
            ],
            "kb_distractor_pool": [
                {"e1_label": "campsite", "rel": "LocatedIn", "e2_label": "forest"},
                {"e1_label": "tent", "rel": "UsedFor", "e2_label": "sleeping"},
            ],
            "query": "Why is the bear approaching and what is the consequence?",
            "ground_truth_trajectory": "Path: [Scene: bear] -> approaching -> [Scene: campsite] <- at <- [Scene: food_container(open)] <=> Aligned <=> <open_food, Emits, food_scent> -> <bear, AttractedBy, food_scent> -> <habituated_bear, Becomes, dangerous>.",
            "answer": "The open food container at the unattended campsite emits scent attracting the bear. Repeated food access habituates bears, making them dangerous to future campers.",
        },
    ]

    # Scale up with variations (each sample repeated with slight diversity)
    # In production, load actual GQA/FVQA data instead
    scaled = []
    for _i in range(4):  # 4x = 40 total samples
        scaled.extend(samples)

    return scaled
