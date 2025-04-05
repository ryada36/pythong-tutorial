# basic class

class Animal:
    def __init__(self, sound):
        self.sound = sound

    def make_sound(self):
        return self.sound


# Inheritance
class Dog(Animal):
    def __init__(self, sound, breed):
        super().__init__(sound)
        self.breed = breed

    def get_breed(self):
        return self.breed
    
# Polymorphism(many forms or use case of same method(make_sound))
class Cat(Animal):
    def __init__(self, sound, color):
        super().__init__(sound)
        self.color = color
    def get_color(self):
        return self.color
    def make_sound(self):
        return f"{self.sound} meow"
    
# my_cat = Cat("Meow", "Black")
# my_dog = Dog("Bark", "Labrador")
# print(my_cat.make_sound())  # Output: Meow meow
# print(my_dog.make_sound())  # Output: Bark

# Encapsulation
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private attribute

    @property
    def balance(self):
        return self.__balance
    
    @balance.setter
    def balance(self, amount):
        if amount >= 0:
            self.__balance = amount
        else:
            print("Balance cannot be negative")

    # def deposit(self, amount):
    #     self.__balance += amount

    # def withdraw(self, amount):
    #     if amount <= self.__balance:
    #         self.__balance -= amount
    #     else:
    #         print("Insufficient funds")

    def get_balance(self):
        return self.__balance
# account = BankAccount(1000)
# # account.deposit(500)
# # account.withdraw(200)
# print(account.balance)  # Output: 1300
# account.balance = -200
# print(account.balance)  # Output: 1300

# print(account.__dict__) 


# sharing attributes among instances

# class Shared:
#     pass
# shared_store = {}

# obj1 = Shared()
# obj2 = Shared()

# obj1.__dict__ = shared_store
# obj2.__dict__ = shared_store

# obj1.name = "John"

# print(obj2.name)  # Output: John


# sharing attributes via metaclass
class SharedMeta(type):
    def __new__(cls,name, bases, attrs):
        attrs["shared_store"] = {}
        return super().__new__(cls, name, bases, attrs)
    
class Shared(metaclass=SharedMeta):
    pass

a = Shared()
b = Shared()

a.shared_store["name"] = "John"
b.shared_store["age"] = 25

print(a.shared_store)  # Output: {'name': 'John', 'age': 25}
       