#The talk() method for this ArnoldBot has been modified so that it takes a message and returns a
#a respone rather than running a while loop that takes in user input at the console and generating
#a response for each user input.

#This is a class that represents ArnoldBot, a Schwarzenegger-themed chatbot.
#This module can be run to speak directly with ArnoldBot.

import random, nltk, time, pickle
from collections import defaultdict
from nltk.corpus import stopwords

class ArnoldBot:        
    def __init__(self):
        '''Initializes the attributes of ArnoldBot.'''
        nonword = "\n" #Need this as a sort of initial placeholder for building the Markov chain.
        self._w1, self._w2, self._w3 = nonword, nonword, nonword
        #^Here we're initializing the very front of the Markov chain with a
        #marker for nothing, essentially.
        n = 5
        self.min_msg_len = n
        #ArnoldBot will say at least n-1 words (1 is reserved for "arnoldbot:" ).
        #Used this^ to prevent fragmenty responses.

        #brain = self._build_brain()
        #self._save_pickle("ArnoldBot_brain", brain)
        self._brain = self._load_pickle("ArnoldBot_brain")

        self._resp = "(blank)"
        self._next_word = "(blank)" #say "(blank)" to prevent string index out of bounds errors.
        self._bad_first_words = ["of", "to", "him", "her", "them", "have", "want", "and",
                                 "if", "myself", "me", "out", "off", "yourself", "myself",
                                 "'em", "at", "is"]

        #freq_trigrams = self._get_freq_trigrams("knowledge.txt")
        #self._save_pickle("freq_trigrams", freq_trigrams)
        self._freq_trigrams = self._load_pickle("freq_trigrams")
        self._stop_words = stopwords.words("english") #The "e" in "english" must be lowercase in order for Heroku deployment to work!

        
    def _save_pickle(self, filename, to_dump):
        '''Takes the name to give a pickle file and the object to dump.'''
        save_brain = open(filename+".pickle", "wb")
        pickle.dump(to_dump, save_brain)
        save_brain.close()
        
        
    def _load_pickle(self, filename):
        '''Takes the name of a pickle file and returns the object from the pickle file.'''
        brain = open(filename + ".pickle", "rb")
        loaded_pickle = pickle.load(brain)
        brain.close()
        return loaded_pickle
      
      
    def _build_brain(self) -> defaultdict: #Private method
        '''Constructs the Markov chain that represents the brain of ArnoldBot.'''
        infile = open(self._training_text)
        content = infile.read()
        infile.close()

        m_chain = defaultdict(list)
        for word in content.split():
            m_chain[(self._w1,self._w2,self._w3)].append(word.lower())
            self._w1, self._w2, self._w3 = self._w2, self._w3, word

        m_chain[(self._w1,self._w2,self._w3)].append("\n")
        return m_chain
      
      
    #Perhaps the frequent trigrams is something that should be obtained upon loading ArnoldBot up!
    #The attribute could be get _freq_trigrams (so rename this _get_freq_trigrams)
    def _get_freq_trigrams(self, text: str) -> list: #Private method  
        '''Determines the most frequently occurring trigrams in a text file.'''
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
      
      
    def _check_state(self, state: tuple) ->bool: #Private method
        '''Takes a state and checks to see if the state exists in the MC.'''
        return state in self._brain.keys()


    def _unravel_freq_tri(self) -> str: #Private method
        '''Unravels the Markov chain from a frequently occurring trigram.'''
        #freq_tris = self._freq_trigrams(self._training_text) #Not doing this anymore since we're pickling
        unwanted = {("a", "lot", "of"), ("out", "of", "the"), ("are", "you", "doing?"), ("to", "be", "a"),
                    ("don't", "want", "to"), ("going", "to", "be"), ("don't", "know", "what"), ("out", "of", "here"),
                    ("want", "to", "be"), ("want", "you", "to"), ("to", "do","with"), ("not", "going", "to"),
                    ("have", "to", "do"), ("of", "a", "bitch!"), ("do", "you", "think")}
        top_50 = {self._freq_trigrams[j][0] for j in range(50)} #This is a set of tuples!
        wanted = top_50.difference(unwanted) #Hmm, probably would have been more efficient to make this an instance variable.
        self._next_word = random.choice(list(wanted)) #Note how next_word is a tuple here. It's a string in other cases.
        self._resp += " " + self._next_word[0] + " " + self._next_word[1] + " " + self._next_word[2]
        self._w1, self._w2, self._w3 = self._next_word[0], self._next_word[1], self._next_word[2]
        

    def _msg_by_keyword(self, lst: list): #Private method
        '''Takes a list of words (in particular, the words in the user's message) and selects a state in the Markov
           Chain that corresponds to a selected keyword from this list of words.'''
        table = str.maketrans("!?.,", "    ")
        no_puncs = [word.translate(table).strip() for word in lst]
        non_sws = [word for word in no_puncs if word not in self._stop_words and word[-3:] != "ing"]
        keyword = ""
        if len(non_sws) != 0: #If there's something in the non-stopwords list...
            lens = [len(e) for e in non_sws]
            keyword = non_sws[lens.index(max(lens))] #pick the longest non stopword cuz it might be the most interesting.
            #related_states is a list holding all states (3-tuples) that are related to the keyword.
            related_states = [key for key in self._brain.keys() if keyword in key and key[0][-1] not in "!?.,"
                              and key[1][-1] not in "!?.,"] #Don't want the first or second component to be an ending word
                                                             #because response could look bad.

            self._select_initial_state_from_related_states(related_states, non_sws, keyword)
        else:
            #No dice. Just pick a frequently occurring trigram.
            self._unravel_freq_tri()
            
            
    def _select_initial_state_from_related_states(self, related_states, non_sws, keyword):
        '''Selects the initial state from which ArnoldBot's message will be unraveled.'''
        if len(related_states) != 0:
            #maybe _next_word isn't such an accurate variable name...
            self._next_word = random.choice(related_states) #self._next_word is a 3-tuple
            just_go_trigram = False #I.e., just go to frequent tri-game mode of answering
            it_limit = 2 #to prevent from possibly bouncing back and forth between keywords (e.g., when len(non_sws) is 2)
            current_it = 0
            #while the first word in the state is a bad first word, pick another keyword.
            while ((self._next_word[0] in self._bad_first_words or self._next_word[0][-3:] == "ing" or
                    self._next_word[0][-2:] == "ed") and current_it < it_limit):
                old_keyword = keyword
                other_keywords = [word for word in non_sws if word != old_keyword]
                if len(other_keywords) == 0:
                    just_go_trigram = True
                    break
                keyword = random.choice(other_keywords)
                related_states = [key for key in self._brain.keys() if keyword in key and key[0][-1] not in ".,!?;:"
                                  and key[1][-1] not in "!?.,"] #Reassigning related_states...
                if len(related_states) == 0:
                    just_go_trigram = True
                    break
                self._next_word = random.choice(related_states)
                current_it += 1

            if just_go_trigram == True:
                self._unravel_freq_tri()
            else:
                self._resp += " " + self._next_word[0] + " " + self._next_word[1] + " " + self._next_word[2]
                self._w1, self._w2, self._w3 = self._next_word[0], self._next_word[1], self._next_word[2]
        else:
            #No dice. Just pick a frequently occurring trigram.
            self._unravel_freq_tri()
            
            
    def _type_out(self, msg: str): #Private method
        '''Prints out a response as though it were being typed out by ArnoldBot.'''
        lst = [word for word in msg.split() if word != "ArnoldBot:"]
        sent = " ".join(lst)
        print("ArnoldBot: ", end = "")
        for l in sent:
            print(l,end = "")
            time.sleep(0.02)
        print("")
        
        
    def _determine_seed_from_3_words(self, words): #Private method
        '''Determines a potential seed with the user's three-worded message (potential because a different one could be selected
           upon beginning the processes of unraveling the Markov Chain.'''
        #If msg_len is 3, then just use those three words.
        self._w1, self._w2, self._w3 = words[0], words[1], words[2]
        return "unravel"
      
      
    def _determine_seed_from_2_words(self, words): #Private method
        '''Determines a potential seed for ArnoldBot's response by using the user's message when the user's message is of
           length 2. Returns either "unravel" if a suitable seed has been acquired or "continue" if the search for another
           seed needs to be carried out.'''
        self._w1, self._w2 = words[0], words[1]
        state_component_exists = len([key[2] for key in self._brain.keys() if \
                                      key[0] == words[0] and key[1] == words[1]])
        outcome = ""

        if state_component_exists:
            self._w3 = random.choice([key[2] for key in self._brain.keys() if \
                                      key[0] == words[0] and key[1] == words[1]])
            outcome = "unravel"
        else: #If the state component doesn't exist...                          
                #First find related states for the longer component. If no related states exist,
                #find related states for the smaller component. If no related states exist, then
                #repeat the user's message and continue.
            lens = [len(e) for e in words] #note how words is the user's message. Here the user's message is two words.
            bigger_word = words[lens.index(max(lens))]
            smaller_word = words[lens.index(max(lens))-1] #good move. if index is 0, then -1. If 1, then 0. Always the other.
            keyword = bigger_word
            related_states = [key for key in self._brain.keys() if keyword in key and key[0][-1] not in ".,!?;:"
                              and key[1][-1] not in ".,!?;:"] #Don't want the first or second component to be the end.

            if len(related_states) != 0:
                sel_state = random.choice(related_states)
                self._w1, self._w2, self._w3 = sel_state[0], sel_state[1], sel_state[2]
                outcome = "unravel"
            else:
                keyword = smaller_word
                related_states = [key for key in self._brain.keys() if keyword in key and key[0][-1] not in ".,!?;:"
                                  and key[1][-1] not in ".,!?;:"] #Don't want the first or second component to be the end.
                if len(related_states) != 0:
                    sel_state = random.choice(related_states)
                    self._w1, self._w2, self._w3 = sel_state[0], sel_state[1], sel_state[2]
                    outcome = "unravel"
                else:
                    self._resp = words[0] + " " + words[1] + "...?"
                    #self._type_out(self._resp) #Don't type out in here if you want talk to return a response string!
                    outcome = "continue"
        return outcome
      
      
    def _determine_seed_from_1_word(self, words): #Private method
        '''Determines a potential seed for ArnoldBot's response by using the user's message when the user's message is
           of length 1. Returns either "unravel" if a suitable seed has been acquired or "continue" if the search for
           another seed needs to be carried out.'''
        self._w1 = words[0]
        state_components_exist = [key for key in self._brain.keys() if key[0] == words[0]]
        outcome = ""
        if state_components_exist:
            rand_state = random.choice([key for key in self._brain.keys() if key[0] == words[0]])
            self._w2 = rand_state[1]
            self._w3 = rand_state[2]
            return "unravel"
        else: #If the state component doesn't exist in the Markov chain...
            self._resp = words[0] + "...?"
            #self._type_out(self._resp) #Don't type out in here if you want talk to return a response string!
            return "continue"
          
          
    def _determine_seed_from_0_words(self):
        '''When 0 words are sent by the user, no seed is determined and a puzzled response will be sent by ArnoldBot.'''
        self._resp = "what do you have to say for yourself?"
        #self._type_out(self._resp) #Don't type out in here if you want talk to return a response string
        return "unravel"
      
      
    def _determine_seed_from_mtt_words(self, words, msg_len): #Private method
        '''Determines a potential seed for ArnoldBot's response by using the user's message when the user's message is
           more than three (mtt) words long. The process of unraveling could very well begin after the selection of this
           potential seed, so 'unravel' will be returned.'''
        w1_index = random.choice(range(msg_len)) #"word 1 index". pick a random index in the split message list
        #Check to see if the randomly selected index is the last index. That's bad.
        if (w1_index > msg_len-3): #If the index would cause the trigram to "fall over the edge"
            w1_index = msg_len-3
            self._w1 = words[w1_index]
            #Shave off punctuation marks if selected trigram isn't at the end of the sentence so the resulting state has
            #a higher chance of being in the MC (training text could be inconsistent with punctuation in that regard).
            self._w2 = words[w1_index+1][:-1] if words[w1_index+1][-1] in ".,!?;:" else words[w1_index+1]
            self._w3 = words[w1_index+2][:-1] if words[w1_index+2][-1] in ".,!?;:" and w1_index+2 != msg_len-1 else words[w1_index+2]
        else: #If the chosen index allows the trigram to fit
            self._w1 = words[w1_index]
            self._w2 = words[w1_index+1][:-1] if words[w1_index+1][-1] in ".,!?;:" else words[w1_index+1]
            self._w3 = words[w1_index+2][:-1] if words[w1_index+2][-1] in ".,!?;:" and w1_index+2 != msg_len-1 else words[w1_index+2]
        return "unravel"
      
      
    def _determine_potential_seed(self, words, msg_len): #Private method
        '''From the user's message, Dermines the initial state (i.e., the seed) of the Markov Chain that will be used in
           building ArnoldBot's response.'''
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
          
          
    def _ensure_no_fitb_situations(self, state_in_MC, words): #Private method
        '''Runs a check to ensure that no fill-in-the-blank situations arise. E.g., User: "How are you?"
           ArnoldBot: "Doing I am good".'''
        if (state_in_MC and (len(words) > 3 and (self._w1 != words[-3] and self._w2 != words[-2] and self._w3 != words[-1]))
            or (len(words) == 2 and (self._w1 == words[0] and self._w2 == words[1])) 
            or (len(words) == 1 and (self._w1 == words[0]))):
                if self._w1 == "am":
                    self._w1 = "I am"
                self._resp += " " + self._w1 + " " + self._w2 + " " + self._w3


    def _unravel(self, words, i): #Private method
        '''Unravels the Markov Chain used to construct ArnoldBot's response.'''
        while True:                
            state_in_MC = self._check_state((self._w1, self._w2, self._w3))

            if (state_in_MC): #If the key is in the MC, just unravel.   
                self._next_word = random.choice(self._brain[(self._w1,self._w2,self._w3)])
                if self._next_word[-1] in ".,?!;" and len(self._resp.split()) < self.min_msg_len:
                    self._resp += " " + self._next_word[:-1] #don't include the punctuation mark yet!!
                else:
                    self._resp += " " + self._next_word

                self._w1, self._w2, self._w3 = self._w2, self._w3, self._next_word
            else: #if the key isn't in the MC, generate a new key according to a keyword in user's message
                #Once you're in here, you're bound to get something that is a valid key.
                self._msg_by_keyword(words)                            
            i += 1
            if self._next_word[-1] in "?!." and len(self._resp.split()) > self.min_msg_len:
                break
        return i
      
      
    def speak(self, msg): #Public method
        '''Initiates a conversation with ArnoldBot.'''
        #msg = ""
        #msg = input("You: ")
        self._resp = "(blank)"

        if ("bye" in msg):
            self._resp = "hasta la vista, baby!"
            #self._resp = "\nArnoldBot: hasta la vista, baby!"
            #self._type_out(self._resp)
            #break
            return self._resp

        #...............Code for determining initial state (seed) of the Markov Chain.................
        else:
            words = [word.lower() for word in msg.split()]
            msg_len = len(words)
            outcome = self._determine_potential_seed(words, msg_len)
            if outcome == "continue":
                    #^ArnoldBot's response would be typed out in _determine_potential_seed is continue was returned
                    #continue
                    return self._resp

        #...............Code for unraveling the Markov Chain...............
        i = 0 #If i is more than zero, then that means the first generated response had a bad form.
        while (i == 0 or self._resp.split()[1] in self._bad_first_words or self._resp.split()[1][-3:] == "ing" or
               self._resp.split()[1][-2:] == "ed"):
            #^This while loop is here to continue building other responses in case a "bad" response is initially built.
            self._resp = ""
            state_in_MC = self._check_state((self._w1, self._w2, self._w3))

            if (i > 0): #If i > 0, then that means the first generated response had a bad form.
                self._msg_by_keyword(words) #words is initially all the words in the user's message

            self._ensure_no_fitb_situations(state_in_MC, words)
            i = self._unravel(words, i)

        #self._type_out(self._resp) #print the response
        return self._resp.strip()
