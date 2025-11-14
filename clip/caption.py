import csv

# Dataset: 4 categories × 5 images each
data = [
    # Beaches
    ["beach1.jpg", "beaches", "A sunny beach white sand, a lot of palm trees and wood structures."],
    ["beach2.jpg", "beaches", "A sunny golden beach with turqoise water and a modern house."],
    ["beach3.jpg", "beaches", "A sunny golden beach and a lot of palm trees."],
    ["beach4.jpg", "beaches", "A beach with white sand, blue water and near of a city."],
    ["beach5.jpg", "beaches", "A beach some rocks, a lot of people and near of a city."],

    # Food
    ["food1.jpg", "food", "A pork chop with salad and nachos."],
    ["food2.jpg", "food", "An hamburger with cheese, lettuce and tomate."],
    ["food3.jpg", "food", "A paella with shrimp and mussels."],
    ["food4.jpg", "food", "A pizza with black olives, meat and peppers."],
    ["food5.jpg", "food", "A bowl with vegetables"],

    # Home interior
    ["home1.jpg", "home_interior", "A modern living-room made of wood with black chairs"],
    ["home2.jpg", "home_interior", "A living-room with a white sofa with wood furniture"],
    ["home3.jpg", "home_interior", "A classic living-room with two sofas and some plants"],
    ["home4.jpg", "home_interior", "A living-room with a black sofa, a table and a kitchen"],
    ["home5.jpg", "home_interior", "A living-room with wood furniture, a sofa and a armchair"],

    # Wild animals
    ["wild1.jpg", "wild_animals", "A lion resting."],
    ["wild2.jpg", "wild_animals", "An elephant walking through the grass."],
    ["wild3.jpg", "wild_animals", "A sitting leopard."],
    ["wild4.jpg", "wild_animals", "A zebra eating dry grass."],
    ["wild5.jpg", "wild_animals", "Three giraffes walking through the jungle."]
]

# Create the CSV file
with open("captions.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "category", "caption"])  # header
    writer.writerows(data)

print("✅ captions.csv created successfully with English category names.")
