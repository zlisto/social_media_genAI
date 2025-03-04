import requests
import time
from datetime import datetime
from elevenlabs import ElevenLabs

class ElevenLabsAPI:
    """
    A wrapper class for interacting with the ElevenLabs Conversational AI API.
    
    This class provides methods to:
    - Retrieve available AI agents
    - Fetch and update agent details
    - Retrieve past conversations and filter them
    """

    def __init__(self, api_key):
        """
        Initialize the ElevenLabs API client.

        Args:
            api_key (str): The ElevenLabs API key for authentication.
        """
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1/convai"
        self.client = ElevenLabs(api_key = api_key)
        self.AGENT_IDS_PROTECTED = []

    def get_agents(self):
        """
        Fetches a list of available ElevenLabs AI agents.

        Returns:
            list: A list of agent dictionaries containing "agent_id" and "name".
        """
        agents = self.client.conversational_ai.get_agents().agents
        agents = [agent for agent in agents if agent.agent_id not in self.AGENT_IDS_PROTECTED ]
        #agents =  [agent for agent in agents ]
        return agents
    
    def get_agent(self, agent_id):
        """
        Retrieves configuration details for a specific ElevenLabs AI agent.

        Args:
            agent_id (str): The ID of the agent.

        Returns:
            dict: A dictionary containing the agent's data:
                - "agent_id": Agent ID
                - "name": Agent name
                - "first_message": The first message the agent sends
                - "max_duration_seconds": Maximum conversation duration
                - "prompt": The conversation prompt
        """

        try:
            agent = self.client.conversational_ai.get_agent(agent_id)

            return agent

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching agent data for {agent_id}: {e}")
            return {"error": str(e)}


    def update_agent(self, data):
        """
        Updates the agent's configuration, including the first message, prompt, and conversation duration.

        Args:
            data (dict): A dictionary that may contain:
                - agent_id (str)
                - name (str)
                - first_message (str)
                - prompt (str)
                - max_duration_seconds (int)
                - llm (str)

        Returns:
            bool: True if the update request was successful (HTTP 200).
        """

        agent_id = data.get("agent_id")
        if not agent_id:
            print("No 'agent_id' provided, cannot update agent.")
            return False

        print(f"Updating agent_id {agent_id}")
        url = f"{self.base_url}/agents/{agent_id}"
        headers = {"xi-api-key": self.api_key}

        # Build the payload conditionally
        payload = {}
        conversation_config = {}
        agent_config = {}
        prompt_config = {}
        conversation_data = {}

        # Top-level name
        if "name" in data:
            payload["name"] = data["name"]

        # agent_config items
        if "first_message" in data:
            agent_config["first_message"] = data["first_message"]

        if "prompt" in data or "llm" in data:
            # Only create the "prompt" sub-dict if we have at least one relevant key
            if "prompt" in data:
                prompt_config["prompt"] = data["prompt"]
            if "llm" in data:
                prompt_config["llm"] = data["llm"]
            if prompt_config:  # If not empty
                agent_config["prompt"] = prompt_config

        if agent_config:
            conversation_config["agent"] = agent_config

        # conversation_data items
        if "max_duration_seconds" in data:
            conversation_data["max_duration_seconds"] = int(data["max_duration_seconds"])

        if conversation_data:
            conversation_config["conversation"] = conversation_data

        if conversation_config:
            payload["conversation_config"] = conversation_config

        # Perform the PATCH request
        #print(f"Payload: {payload}")
        response = requests.patch(url, json=payload, headers=headers)
        return response.status_code == 200



    def get_all_conversations(self, agent_id):
        """
        Retrieves all conversations for a given AI agent.

        Args:
            agent_id (str): The ID of the agent whose conversations to fetch.

        Returns:
            list: A list of conversation objects.
        """
        all_conversations = []
        has_more = True
        cursor = None  # Initialize cursor for pagination

        while has_more:
            # Make API call with or without cursor
            if cursor:
                response = self.client.conversational_ai.get_conversations(agent_id=agent_id, cursor=cursor)
            else:
                response = self.client.conversational_ai.get_conversations(agent_id=agent_id)

            # Append retrieved conversations
            all_conversations.extend(response.conversations)

            # Check if there's more data to fetch
            has_more = response.has_more
            cursor = response.next_cursor if has_more else None

            time.sleep(0.1)  # Prevent API rate limiting

        return all_conversations


    def get_conversation(self, conversation_id):
        """
        Fetches the details of a specific conversation.

        Args:
            conversation_id (str): The unique ID of the conversation.

        Returns:
            dict: The conversation details.
        """
        response = self.client.conversational_ai.get_conversation(conversation_id)
        return response
    def get_most_recent_conversation(self, agent_id):
        """
        Retrieves the most recent conversation for a given AI agent.

        Args:
            agent_id (str): The ID of the agent.

        Returns:
            dict: The most recent conversation object.
        """
        all_conversations = self.get_all_conversations(agent_id)
        most_recent_conversation = all_conversations[0]
        conversation_id = most_recent_conversation.conversation_id
        conversation = self.get_conversation(conversation_id)
        return conversation
    def get_most_recent_conversation_string(self,agent_id):
        """
        Retrieves the most recent conversation for a given ElevenLabs agent_d and formats it as a string.
        with format "Most recent conversation between {agent_name} and {speaker_name}\nTIME: {formatted_time}\nCONVERSATION\n{role}: {message}"
        Args:
            agent_id (str): The ID of the agent.
        Returns:
            str: The most recent conversation formatted as a string.
        """
        conversation =self.get_most_recent_conversation(agent_id)
        formatted_time = datetime.fromtimestamp(conversation.metadata.start_time_unix_secs).strftime("%B %d, %Y, %I:%M %p")
        call_duration_secs  = conversation.metadata.call_duration_secs
        agent_name = self.get_agent(agent_id).name
        #speaker_name = conversation.analysis.data_collection_results['SPEAKER'].value
        conversation_transcript  = f"Most recent conversation of {agent_name}\nTIME: {formatted_time}\nCONVERSATION\n"
        for msg in conversation.transcript:
            if msg.role == "agent":
                role = agent_name
            else:
                role = msg.role
            if msg.message != None:
                conversation_transcript += f"{role}: {msg.message}\n"
        return conversation_transcript
            
    def get_conversation_summaries_string(self, agent_id, start_time, call_duration_min_secs):      
        """
        Retrieves all conversations after a given start time and with a minimum call duration, and formats them as a string.
        with format "TIME: {formatted_time}, SPEAKER: {speaker_name}, DURATION: {call_duration_secs} seconds, SUMMARY: {transcript_summary}"
        Args:
            agent_id (str): The ID of the agent.
            start_time (datetime): The start time to filter conversations.
            call_duration_min_secs (int): The minimum call duration in seconds. 
        Returns:
            str: The conversation summaries formatted as a string.
        """

        start_time_unix_sec = int(start_time.timestamp())
        conversations = self.get_all_conversations(agent_id)
        conversations = [conversation for conversation in conversations if conversation.start_time_unix_secs >= start_time_unix_sec]
        conversations = [conversation for conversation in conversations if conversation.call_duration_secs > call_duration_min_secs]
        print(f"\tThere are {len(conversations)} conversations after {start_time} with a minimum call duration of {call_duration_min_secs} seconds.")

        summaries = "["
        for c in conversations:
            tstart = datetime.fromtimestamp(c.start_time_unix_secs)
            conversation = self.get_conversation(c.conversation_id)
            speaker = conversation.analysis.data_collection_results['SPEAKER'].value
            summaries += "\n{" + f"\nTIME: {tstart},\nSPEAKER: {speaker},\nDURATION: {c.call_duration_secs} seconds,\nSUMMARY:{conversation.analysis.transcript_summary}" + "},"
        summaries += "]"
        return summaries

    