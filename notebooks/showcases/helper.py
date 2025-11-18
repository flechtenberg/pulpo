import brightway2 as bw

def create_or_update_activity(db, activity_code, activity_details, exchanges, biosphere):
    """
    Creates or updates an activity with specified technosphere and biosphere exchanges.

    Parameters:
    - db: The Brightway2 database object.
    - activity_code: Unique code for the activity.
    - activity_details: Dictionary containing the name, unit, and location of the activity.
    - exchanges: List of dictionaries detailing the exchanges to be added or updated.
    - biosphere: The Brightway2 biosphere database object for biosphere exchanges.
    """
    try:
        # Try to get the activity if it already exists
        activity = db.get(activity_code)
        print(f"{activity_details['name']} already exists, updating exchanges.")
    except:
        # If not, create a new activity
        activity = db.new_activity(code=activity_code, **activity_details)
        activity.save()
        print(f"Created {activity_details['name']}.")

    # Add or update exchanges
    for exch in exchanges:
        # Determine if the exchange is technosphere or biosphere
        if exch['type'] == 'biosphere':
            search_result = biosphere.search(exch['name'])[0]
        else:  # 'technosphere'
            search_result = db.search(exch['name'])[0]

        # Add or update the exchange
        activity.new_exchange(input=search_result.key, amount=exch['amount'], type=exch['type']).save()
    
    # Save changes to the activity
    activity.save()
    print(f"Updated {activity_details['name']} with new exchanges.")

def copy_activity(original_activity):
    # Construct the new name for the copied activity
    new_activity_name = original_activity['name'] + " copy"

    # Get the database of the original activity
    db = bw.Database(original_activity['database'])

    # Check if the activity with the new name already exists
    existing_activities = {a['name']: a for a in db}
    if new_activity_name in existing_activities:
        print(f"Activity '{new_activity_name}' already exists.")
        return existing_activities[new_activity_name]
    else:
        # Create a new activity as a copy of the original activity
        copied_activity = original_activity.copy()
        copied_activity['name'] = new_activity_name
        copied_activity.save()
        print(f"Created a copy of '{original_activity['name']}' named '{new_activity_name}'.")
        return copied_activity
