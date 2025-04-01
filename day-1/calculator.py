x = input("Enter first number: ")
y = input("Enter second number: ")
operation = input("Enter operation (+, -, *, /): ")
if operation == "+":
    result = float(x) + float(y)
elif operation == "-":
    result = float(x) - float(y)
elif operation == "*":
    result = float(x) * float(y)
elif operation == "/":
    result = float(x) / float(y)
else:
    result = "Invalid operation"
print(f"The result of {x} {operation} {y} is: {result:,.2f}")