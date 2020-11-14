def get_user_answer_boolean(prompt):
    print()
    user_answer = input(prompt).upper()
    
    while user_answer != 'Y' and user_answer != 'N':
        print("Invalid input!")
        user_answer = input(prompt)
        
    return user_answer == 'Y'