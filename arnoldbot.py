#The speak() method for this ArnoldBot has been modified so that it takes a message and returns a
#a respone rather than running a while loop that takes in user input at the console and generating
#a response for each user input.

#This is a class that represents ArnoldBot, a Schwarzenegger-themed chatbot.
#This module can be run to speak directly with ArnoldBot.

import random, nltk, time, pickle
from collections import defaultdict
from nltk.corpus import stopwords




class ArnoldBot:
    """
    A class that represents an Arnold Schwarzenegger
    themed chat bot.
    """

    def __init__(self):
        """ Initializes the attributes of ArnoldBot. """
        # Need this nonword varialbe as a sort of initial
        # placeholder for building the Markov chain.
        nonword = "\n"
        self._w1, self._w2, self._w3 = nonword, nonword, nonword
        #^ Here we're initializing the very front of the Markov chain with a
        #  marker for nothing, essentially.
        n = 5
        self.min_msg_len = n
        # ArnoldBot will say at least n-1 words (1 is reserved for "arnoldbot:" ).
        # Used this^ to prevent fragmenty responses.

        #brain = self._build_brain()
        #self._save_pickle("ArnoldBot_brain", brain)
        self._brain = self._load_pickle("ArnoldBot_brain")
        # Say "(blank)" to prevent string index out of bounds errors:
        self._resp = "(blank)"
        self._next_word = "(blank)"
        self._bad_first_words = ["of", "to", "him", "her", "them", "have",
                                 "want", "and", "if", "myself", "me", "out",
                                 "off", "yourself", "myself", "'em", "at", "is"]

        #freq_trigrams = self._get_freq_trigrams("knowledge.txt")
        #self._save_pickle("freq_trigrams", freq_trigrams)
        self._freq_trigrams = self._load_pickle("freq_trigrams")
        self._stop_words = stopwords.words("english")
        # The "e" in "english" must be lowercase in order
        # for Heroku deployment to work!


    def _save_pickle(self, filename, to_dump):
        """
        Takes the name to give a pickle file and the object to dump.
        @param filename: The name for the .pickle file
        @param to_dump: The object to pickle
        type filename: str
        type to_dump: Any Python object
        return: None
        rtype: None
        """
        save_brain = open(filename+".pickle", "wb")
        pickle.dump(to_dump, save_brain)
        save_brain.close()


    def _load_pickle(self, filename):
        """
        Takes the name of a pickle file and returns the
        object from the pickle file.
        @param filename: The name for the .pickle file
        type filename: str
        return: The Python object loaded from the
                .pickle file
        rtype: Any Python object
        """
        brain = open(filename + ".pickle", "rb")
        loaded_pickle = pickle.load(brain)
        brain.close()
        return loaded_pickle


    def _build_brain(self):
        """
        Constructs the Markov chain that represents the brain of ArnoldBot.
        return: A defaultdict representing the Markov chain. The keys of the
                defaultdict are all trigrams of the training text and the
                values are lists whose elements are strings that appear after
                the trigram key.
        rtype: defaultdict
        """
        infile = open(self._training_text)
        content = infile.read()
        infile.close()

        m_chain = defaultdict(list)
        for word in content.split():
            m_chain[(self._w1,self._w2,self._w3)].append(word.lower())
            self._w1, self._w2, self._w3 = self._w2, self._w3, word

        m_chain[(self._w1,self._w2,self._w3)].append("\n")
        return m_chain


    # Perhaps the frequent trigrams is something that could be obtained
    # upon loading ArnoldBot...Consider using this function for that.
    def _get_freq_trigrams(self, text):
        """
        Determines the most frequently occurring trigrams in a text file.
        @param text: The name of the training text to determine frequently
                     occurring trigrams of
        type text: str
        return: A list of tuples whose first component is a trigram and
                whose second component is its count. The list is sorted
                in descending order by trigram count.
        rtype: list
        """
        infile = open(text)
        content = infile.read()
        words = content.split()
        tri_gs = list(nltk.trigrams(words))
        freq_d = defaultdict(int)

        for triple in tri_gs:
            freq_d[triple] += 1

        counts = list(freq_d.items())
        counts.sort(key = lambda x: x[1], reverse = True)
        return counts


    def _check_state(self, state):
        """
        Takes a state and checks to see if the state exists in the MC.
        @param state: A key (i.e., a trigram of the training text) to
                      check for the existence of in the defaultdict
                      that represents ArnoldBot's brain.
        type state: tuple
        return: True if the key is in self._brain or False otherwise
        rtype: bool
        """
        return state in self._brain.keys()


    def _unravel_freq_tri(self):
        """
        Unravels the Markov chain from a frequently occurring trigram.
        return: None
        rtype: None
        """
        #freq_tris = self._freq_trigrams(self._training_text)
        # Not doing this^ anymore since we're pickling
        unwanted = {("a", "lot", "of"), ("out", "of", "the"), ("are", "you", "doing?"),
                    ("to", "be", "a"), ("don't", "want", "to"), ("going", "to", "be"),
                    ("don't", "know", "what"), ("out", "of", "here"), ("want", "to", "be"),
                    ("want", "you", "to"), ("to", "do","with"), ("not", "going", "to"),
                    ("have", "to", "do"), ("of", "a", "bitch!"), ("do", "you", "think")}

        # NOTE: The following is a set of tuples!
        top_50 = {self._freq_trigrams[j][0] for j in range(50)}
        # Hmm, it probably would have been more efficient to
        # make this an instance variable:
        wanted = top_50.difference(unwanted)

        # Note how next_word is a tuple here. It's a string in other cases...
        self._next_word = random.choice(list(wanted))
        self._resp += " " + self._next_word[0] + " " +\
                      self._next_word[1] + " " + self._next_word[2]

        self._w1, self._w2, self._w3 = self._next_word[0], self._next_word[1], self._next_word[2]


    def _msg_by_keyword(self, lst):
        """
        Takes a list of words (in particular, the words in the user's message)
        and selects a state in the Markov Chain that corresponds to a selected
        keyword from this list of words.
        @param lst: The user's message represented as a list of words
        type lst: [str]
        return: None
        rtype: None
        """
        table = str.maketrans("!?.,", "    ")
        no_puncs = [ word.translate(table).strip() for word in lst ]
        non_sws = [ word for word in no_puncs if word not in\
                    self._stop_words and word[-3:] != "ing" ]
        keyword = ""

        # If there's something in the non-stopwords list...
        if len(non_sws) != 0:
            lens = [len(e) for e in non_sws]
            # Pick the longest non stopword because it might be the most interesting:
            keyword = non_sws[lens.index(max(lens))]
            # related_states is a list holding all states
            # (i.e., 3-tuples) that are related to the keyword.
            related_states = [ key for key in self._brain.keys() if keyword in\
                               key and key[0][-1] not in "!?.," and key[1][-1]\
                               not in "!?.,"]
            # Don't want the first or second component to be an
            # ending word because response could look bad.
            self._select_initial_state_from_related_states(related_states, non_sws, keyword)
        else:
            #No dice. Just pick a frequently occurring trigram.
            self._unravel_freq_tri()


    def _select_initial_state_from_related_states(self, related_states, non_sws, keyword):
        """
        Selects the initial state from which ArnoldBot's message will be unraveled.
        @param related_states: A list containing all states (i.e., 3-tuples)
                               that are related to the provided keyword
        @param non_sws: A list of non-stopwords
        @param keyword: The potentially most interesting word in
                        a message sent to ArnoldBot
        type related_states: [tuple]
        type non_sws: [str]
        type keyword: str
        return: None
        rtype: None
        """
        if len(related_states) != 0:
            # NOTE: self._next_word is a 3-tuple
            # - Maybe _next_word isn't such an accurate variable name...
            self._next_word = random.choice(related_states)
            # just_go_trigram as in just go to a frequent
            # trigram mode of answering
            just_go_trigram = False
            # iter_limit is for preventing from possibly bouncing back
            # and forth between keywords (e.g., when len(non_sws) is 2)
            iter_limit = 2
            current_iter = 0

            # While the first word in the state is a bad first word,
            # pick another keyword.
            while ((self._next_word[0] in self._bad_first_words or\
                    self._next_word[0][-3:] == "ing" or\
                    self._next_word[0][-2:] == "ed") and\
                    current_iter < iter_limit):

                old_keyword = keyword
                other_keywords = [word for word in non_sws if word != old_keyword]
                if len(other_keywords) == 0:
                    just_go_trigram = True
                    break
                keyword = random.choice(other_keywords)
                related_states = [ key for key in self._brain.keys() if keyword\
                                   in key and key[0][-1] not in ".,!?;:" and\
                                   key[1][-1] not in "!?.," ]

                if len(related_states) == 0:
                    just_go_trigram = True
                    break
                self._next_word = random.choice(related_states)
                current_iter += 1

            if just_go_trigram:
                self._unravel_freq_tri()
            else:
                # TODO: Consider putting following two assignment
                #       statements into a helper method...
                self._resp += " " + self._next_word[0] + " " +\
                              self._next_word[1] + " " + self._next_word[2]
                self._w1, self._w2, self._w3 = self._next_word[0], self._next_word[1], self._next_word[2]
        else:
            #No dice. Just pick a frequently occurring trigram.
            self._unravel_freq_tri()


    def _type_out(self, msg):
        """
        Prints out a response as though it were being typed out by ArnoldBot.
        @param msg: The message to "type out"
        type msg: str
        return: None
        rtype: None
        """
        lst = [word for word in msg.split() if word != "ArnoldBot:"]
        sent = " ".join(lst)
        print("ArnoldBot: ", end = "")
        for l in sent:
            print(l,end = "")
            time.sleep(0.02)
        print("")


    def _determine_seed_from_3_words(self, words):
        """
        Determines a potential seed with the user's three-worded message
        (potential because a different one could be selected upon beginning
        the processes of unraveling the Markov Chain).
        @param words: The user's 3-worded messages represented as
                      a list of strings
        type words: [str]
        return: The string "unravel" to indicate the process of
                unraveling ArnoldBot's Markov Chain to the
                calling method
        rtype: str
        """
        #If msg_len is 3, then just use those three words.
        self._w1, self._w2, self._w3 = words[0], words[1], words[2]
        return "unravel"


    def _determine_seed_from_2_words(self, words):
        """
        Determines a potential seed for ArnoldBot's response by using
        the user's message when the user's message is of length 2.
        Returns either "unravel" if a suitable seed has been acquired
        or "continue" if the search for another seed needs to be carried out.
        @param words: The user's 2-worded messages represented as
                      a list of strings
        type words: [str]
        return: "unravel" if a suitable seed has been acquired or
                "continue" if the search for another seed needs to
                be carried out.
        rtype: str
        """
        self._w1, self._w2 = words[0], words[1]
        state_component_exists = len([key[2] for key in self._brain.keys() if \
                                      key[0] == words[0] and key[1] == words[1]])
        outcome = ""

        if state_component_exists:
            self._w3 = random.choice([key[2] for key in self._brain.keys() if \
                                      key[0] == words[0] and key[1] == words[1]])
            outcome = "unravel"
        else:
            # If the state component doesn't exist...
            # First find related states for the longer component.
            # If no related states exist, find related states for
            # the smaller component. If no related states exist,
            # then repeat the user's message and continue.

            lens = [len(e) for e in words]
            bigger_word = words[lens.index(max(lens))]
            # Good move here. if index is 0, then -1.
            # If 1, then 0. Always the other.
            smaller_word = words[lens.index(max(lens))-1]
            keyword = bigger_word
            related_states = [ key for key in self._brain.keys() if keyword\
                               in key and key[0][-1] not in ".,!?;:" and\
                               key[1][-1] not in ".,!?;:"]
            # Don't want the first or second component to be the end here^

            if len(related_states) != 0:
                # TODO: Wrap the following two lines in a helper method
                sel_state = random.choice(related_states)
                self._w1, self._w2, self._w3 = sel_state[0], sel_state[1], sel_state[2]
                outcome = "unravel"
            else:
                keyword = smaller_word
                related_states = [ key for key in self._brain.keys() if\
                                   keyword in key and key[0][-1] not in\
                                   ".,!?;:" and key[1][-1] not in ".,!?;:"]
                # Don't want the first or second component to be the end here^
                if len(related_states) != 0:
                    # TODO: Wrap the following two lines in a helper method
                    sel_state = random.choice(related_states)
                    self._w1, self._w2, self._w3 = sel_state[0], sel_state[1], sel_state[2]
                    outcome = "unravel"
                else:
                    self._resp = words[0] + " " + words[1] + "...?"
                    #self._type_out(self._resp)
                    # Don't type out in here^ if you want speak() to
                    # return a response string!
                    outcome = "continue"
        return outcome


    def _determine_seed_from_1_word(self, words):
        """
        Determines a potential seed for ArnoldBot's response by
        using the user's message when the user's message is of
        length 1. Returns either "unravel" if a suitable seed has
        been acquired or "continue" if the search for another seed
        needs to be carried out.
        @param words: The user's 1-worded message represented as
                      a list of single string
        type words: [str]
        return: "unravel" if a suitable seed has been acquired or
                "continue" if the search for another seed needs to
                be carried out.
        rtype: str
        """
        self._w1 = words[0]
        state_components_exist = [ key for key in self._brain.keys() if key[0] == words[0] ]
        outcome = ""
        if state_components_exist:
            rand_state = random.choice([ key for key in self._brain.keys() if key[0] == words[0] ])
            self._w2 = rand_state[1]
            self._w3 = rand_state[2]
            return "unravel"
        else:
            self._resp = words[0] + "...?"
            #self._type_out(self._resp)
            # Don't type out in here^ if you want speak()
            # to return a response string!
            return "continue"


    def _determine_seed_from_0_words(self):
        """
        When 0 words are sent by the user, no seed is determined and
        a puzzled response will be sent by ArnoldBot.
        return: The string "unravel" to indicate the process of
                unraveling ArnoldBot's Markov Chain to the
                calling method
        rtype: str
        """
        self._resp = "what do you have to say for yourself?"
        #self._type_out(self._resp)
        # Don't type out in here^ if you want speak()
        # to return a response string
        return "unravel"


    def _determine_seed_from_mtt_words(self, words, msg_len):
        """
        Determines a potential seed for ArnoldBot's response by
        using the user's message when the user's message is more
        than three (mtt) words long. The process of unraveling could
        very well begin after the selection of this potential seed,
        so "unravel" will be returned.
        @param words: The user's message represented
                      as a list of strings
        @param msg_len: The length of the user's message
        type words: [str]
        type msg_len: int
        return: The string "unravel" to indicate the process of
                unraveling ArnoldBot's Markov Chain to the
                calling method
        rtype: str
        """
        # TODO: Figure out why you had to pass around a msg_len variable
        #       instead of just saying len(words)...words actually probably
        #       only contains a list of significant words (i.e., non-stopwords).

        # w1_index => "word 1 index"
        w1_index = random.choice(range(msg_len))

        # If the index would cause the trigram to "fall over the edge"...
        if (w1_index > msg_len-3):
            w1_index = msg_len-3
            self._w1 = words[w1_index]
            # Shave off punctuation marks if selected trigram isn't at the end
            # of the sentence so the resulting state has a higher chance of
            # being in the MC (training text could be inconsistent with punc-
            # tuation in that regard).
            self._w2 = words[w1_index+1][:-1] if words[w1_index+1][-1]\
                       in ".,!?;:" else words[w1_index+1]
            self._w3 = words[w1_index+2][:-1] if words[w1_index+2][-1]\
                       in ".,!?;:" and w1_index+2 != msg_len-1 else words[w1_index+2]
        # If the chosen index allows the trigram to fit...
        else:
            self._w1 = words[w1_index]
            self._w2 = words[w1_index+1][:-1] if words[w1_index+1][-1]\
                       in ".,!?;:" else words[w1_index+1]
            self._w3 = words[w1_index+2][:-1] if words[w1_index+2][-1]\
                       in ".,!?;:" and w1_index+2 != msg_len-1 else words[w1_index+2]

        return "unravel"


    def _determine_potential_seed(self, words, msg_len):
        """
        From the user's message, determines the initial state
        (i.e., the seed) of the Markov Chain that will be used in
        building ArnoldBot's response.
        @param words: The user's message represented
                      as a list of strings
        @param msg_len: The length of the user's message
        type words: [str]
        type msg_len: int
        return: An outcome represented by a string that indicates
                the action to be carried out on ArnoldBot's Markov
                Chain
        rtype: str
        """
        if msg_len == 3:
            return self._determine_seed_from_3_words(words)
        elif msg_len == 2: #Case where msg_len is two.
            return self._determine_seed_from_2_words(words)
        elif msg_len == 1:
            return self._determine_seed_from_1_word(words)
        elif msg_len == 0:
            return self._determine_seed_from_0_words()
        else: #So if the msg_len is greater than 3...
            return self._determine_seed_from_mtt_words(words, msg_len)


    # TODO: Come up with a better name than state_in_MC?
    def _ensure_no_fitb_situations(self, state_in_MC, words):
        """
        Runs a check to ensure that no fill-in-the-blank
        situations arise. E.g., User: "How are you?"
        ArnoldBot: "Doing I am good".
        @param state_in_MC: A boolean indicating whether
                            or not some key existed in
                            ArnoldBot's brain
        @param words: The user's message represented
                      as a list of strings
        type state_in_MC: bool
        type words: [str]
        return: None
        rtype: None
        """
        if (state_in_MC and (len(words) > 3 and (self._w1 != words[-3] and
            self._w2 != words[-2] and self._w3 != words[-1])) or
            (len(words) == 2 and (self._w1 == words[0] and self._w2 == words[1]))
            or (len(words) == 1 and (self._w1 == words[0]))):
                if self._w1 == "am":
                    self._w1 = "I am"
                self._resp += " " + self._w1 + " " + self._w2 + " " + self._w3


    def _unravel(self, words, i):
        """
        Unravels the Markov Chain used to construct ArnoldBot's response.
        @param words: The user's message represented
                      as a list of strings
        @param i: A loop control variable used by the while loop
                  in which this method is called inside of
        type words: [str]
        type i: int
        return: A modified loop control variable
        rtype: int
        """
        while True:
            state_in_MC = self._check_state((self._w1, self._w2, self._w3))

            # If the key is in the MC, just unravel...
            if (state_in_MC):
                self._next_word = random.choice(self._brain[(self._w1,self._w2,self._w3)])
                if self._next_word[-1] in ".,?!;" and len(self._resp.split()) < self.min_msg_len:
                    # Don't include the punctuation mark yet with this:
                    self._resp += " " + self._next_word[:-1]
                else:
                    self._resp += " " + self._next_word

                self._w1, self._w2, self._w3 = self._w2, self._w3, self._next_word
            # Else, the key isn't in the MC, generate a new key
            # according to a keyword in user's message
            else:
                # Once you're in here, you're bound to get
                # something that is a valid key.
                self._msg_by_keyword(words)
            i += 1
            if self._next_word[-1] in "?!." and len(self._resp.split()) > self.min_msg_len:
                break

        return i


    def speak(self, msg):
        """
        Initiates a conversation with ArnoldBot.
        @param msg: The user's message to ArnoldBot
        type msg: str
        return: A response to the user's message
        rtype: str
        """
        #msg = ""
        #msg = input("You: ")
        self._resp = "(blank)"

        if ("bye" in msg):
            self._resp = "hasta la vista, baby!"
            #self._resp = "\nArnoldBot: hasta la vista, baby!"
            #self._type_out(self._resp)
            #break
            return self._resp

        # ===== Code for determining initial state (seed) of the Markov Chain =====
        else:
            words = [word.lower() for word in msg.split()]
            msg_len = len(words)
            outcome = self._determine_potential_seed(words, msg_len)
            if outcome == "continue":
                    # ^ArnoldBot's response would be typed out in
                    # _determine_potential_seed() is "continue" was returned
                    #continue
                    return self._resp

        # ===== Code for unraveling the Markov Chain =====
        # NOTE: If i is more than zero, then that means the
        #       first generated response had a bad form.
        i = 0
        # Seems like this i variable was a workaround of some sort
        # to encapsulate the code found in _unravel()...
        while (i == 0 or self._resp.split()[1] in self._bad_first_words
               or self._resp.split()[1][-3:] == "ing" or
               self._resp.split()[1][-2:] == "ed"):
            # ^This while loop is here to continue building other
            # responses in case a "bad" response is initially built.
            self._resp = ""
            state_in_MC = self._check_state((self._w1, self._w2, self._w3))

            # If i > 0 at this point, then that means
            # the first generated response had a bad form.
            if (i > 0):
                # NOTE: words is initially all the words in the user's message
                self._msg_by_keyword(words)

            self._ensure_no_fitb_situations(state_in_MC, words)
            i = self._unravel(words, i)

        #self._type_out(self._resp) #print the response
        return self._resp.strip()
