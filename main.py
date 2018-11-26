import rnn_character as rnn_rand
import rnn_characters_pretrained as rnn_pretrained
import rnn_character_entitylibrary as rnn_entity


######################### Run the RNN-rand model  #######################
rnn_rand.rnn_character("AskAPatient",train_model=False)
rnn_rand. rnn_character("TwADR-L",train_model=False)

######################### Run the RNN-pretrained model  #######################
rnn_pretrained.rnn_characters_pretrained("AskAPatient", train_model=False)
rnn_pretrained.rnn_characters_pretrained("TwADR-L",train_model=False)

######################### Run the RNN-entity model  #######################
rnn_entity.rnn_character_entity("AskAPatient",train_model=True)
rnn_entity.rnn_character_entity("TwADR-L",train_model=True)