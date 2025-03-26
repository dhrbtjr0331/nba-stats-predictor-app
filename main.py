from models.predict import Predictor

def main():
    """Main function that takes user input and provides prediction output"""
    predictor = Predictor()

    # User input
    player_name = input("Enter Player Name: ")
    opponent_team = input("Enter Opponent Team: ")
    home_or_away = input("Enter Home/Away: ")

    # Make prediction
    predictions = predictor.predict(player_name, opponent_team, home_or_away)

    if predictions is not None:
        print("\n📊 Predicted Stats:")
        print(predictions)

if __name__ == "__main__":
    main()