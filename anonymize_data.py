import json
import os


def anonymize_value(value):
    """
    Determines the anonymized value based on the type of the original value.
    """
    if isinstance(value, str):
        return "ANONYMIZED_STRING"
    elif isinstance(value, (int, float)):
        return 0
    elif isinstance(value, bool):
        # Or True, depending on desired anonymization for booleans
        return False
    elif isinstance(value, list):
        return []
    elif isinstance(value, dict):
        return {}
    else:
        return "ANONYMIZED"


def anonymize_data(data, keys_to_anonymize):
    """
    Recursively anonymizes specified keys in a dictionary or list.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_anonymize:
                data[key] = anonymize_value(value)
            else:
                anonymize_data(value, keys_to_anonymize)
    elif isinstance(data, list):
        for item in data:
            anonymize_data(item, keys_to_anonymize)
    return data


def main():
    data_folder = "data"
    # Add all the keys you want to anonymize here. Case-sensitive.
    keys_to_anonymize = [
        # General PII - common variations
        "id", "ID", "Id",
        "userId", "user_id", "userID", "User_ID",
        "profileId", "profile_id", "ProfileID", "Profile_ID",
        "username", "userName", "user_name",
        "name", "Name", "fullName", "full_name",
        "firstName", "first_name", "FirstName", "First_Name",
        "lastName", "last_name", "LastName", "Last_Name",
        "email", "Email", "emailAddress", "email_address",
        "phone", "Phone", "phoneNumber", "phone_number",
        "address", "Address", "street", "streetAddress", "street_address",
        "city", "City", "state", "State", "zipCode", "zip_code",
        "postalCode", "postal_code", "country", "Country",
        "dateOfBirth", "dob", "birthDate", "birthdate",
        "ssn", "socialSecurityNumber", "nationalId",

        # Keys specifically mentioned by the user from paths like:
        # dailySleepDTO.userProfilePK
        # dailySleepDTO.sleepNeed.userProfilePk
        # wellnessEpochSPO2DataDTOList[*].deviceId
        "userProfilePK",  # From dailySleepDTO.userProfilePK
        "userProfilePk",  # From dailySleepDTO.sleepNeed.userProfilePk
        "deviceId",       # From dailySleepDTO.sleepNeed.deviceId

        # Add any other sensitive keys below
    ]  # Add more as needed

    # Create a subfolder for anonymized data
    anonymized_folder = os.path.join(data_folder, "anonymized")
    if not os.path.exists(anonymized_folder):
        os.makedirs(anonymized_folder)

    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            anonymized_file_path = os.path.join(anonymized_folder, filename)

            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Skipping.")
                continue

            anonymized_content = anonymize_data(content, keys_to_anonymize)

            try:
                with open(anonymized_file_path, 'w') as f:
                    json.dump(anonymized_content, f, indent=4)
                print(f"Anonymized data saved to {anonymized_file_path}")
            except Exception as e:
                error_message = (
                    f"Error writing anonymized data to "
                    f"{anonymized_file_path}: {e}"
                )
                print(error_message)


if __name__ == "__main__":
    main()