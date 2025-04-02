# dictionaries
person = {
    "name": "John",
    "age": 30,
    "is_student": False,
}
print(f"{person["name"]} is {person["age"]} years old.")

# lists (can contain different data types and duplicates)
fruits = ["apple", "banana", "cherry"];
print(fruits)  # all fruits
print(fruits[1:3])

fruits.append("orange")  # add an item
print(fruits)  # all fruits
fruits.sort()  # sort the list
print(fruits)  # sorted fruits
fruits.reverse()  # reverse the list
print(fruits)  # reversed fruits

# immutability
m_fruits = sorted(fruits)
print(m_fruits)
spliced_fruits = m_fruits[1:3]
print(spliced_fruits)
print(m_fruits)  # original list is unchanged

# sets
my_set = {1, 2, 3, 4, 5}
print(my_set)
my_set.add(6)  # add an item
print(my_set)
my_set.remove(2)  # remove an item
print(my_set)
unioned_set = my_set.union({7, 8, 9})
print(unioned_set)
# intersection
intersected_set = my_set.intersection({4, 5, 8})
print(intersected_set)
# create set from list
set_from_list = set(fruits)
set_from_list.add("banana")  # add an existing item
print(set_from_list)

# tuples : immutable,faster than list, dictionary keys(co-ordinates)
coordinates = {(10,20): "Park", (30,39): "Mall"}
print(coordinates[(10,20)])
# return multiple values from function
def give_park_location():
    for coord, place in coordinates.items():
        if place == 'Park':
            return coord
        

park_coord = give_park_location()
print(park_coord)

# function with ternary condition
def check_ternary(val: int):
    return True if val > 1  else False

print(check_ternary(0))


# function with switch case
def switcher(val):
    match(val):
        case 1:
            return "One"
        case 2:
            return "Two"
        case _:
            return "Unknown"
        
print(switcher(2))
    
    
    
