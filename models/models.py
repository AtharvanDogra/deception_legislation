from guidance.models.transformers import TransformersChat
from guidance import models

# Not being used as swtiched to llama.cpp

# class LlamaChat(TransformersChat):
#     def get_role_start(self, role_name, **kwargs):
#         if role_name == "user":

#             # if we follow an auto-nested system role then we are done
#             if self._current_prompt().endswith("\n<</SYS>>\n\n"):
#                 return ""
#             else:
#                 return "[INST] "
        
#         elif role_name == "assistant":
#             return " "
        
#         elif role_name == "system":
            
#             # check if we are already embedded at the top of a user role
#             if self._current_prompt().endswith("[INST] "):
#                 return "<<SYS>>\n"

#             # if not then we auto nest ourselves
#             else:
#                 return "[INST] <<SYS>>\n"
    
#     def get_role_end(self, role_name=None):
#         if role_name == "user":
#             return " [/INST]"
#         elif role_name == "assistant":
#             return " "
#         elif role_name == "system":
#             return "\n<</SYS>>\n\n"
        
class QwenChat(models.llama_cpp.LlamaCpp, models.Chat):
    
    def get_role_end(self, role_name=None):
        '''The ending bytes for a role.
        
        Different from the guidance chat implementations, adding a extra '\n' at the end,
        following tokenizer.apply_chat_template for Qwen/Qwen1.5-72B-Chat
        
        Parameters
        ----------
        role_name : str
            The name of the role, like "system", "user", or "assistant"
        '''
        return "<|im_end|>\n"
    
class Yi(models.llama_cpp.LlamaCpp, models.Chat):
    
    def get_role_start(self, role_name, **kwargs):
        if role_name == "system":
            return ""
        else:
            return "<|im_start|>"+role_name+"".join([f' {k}="{v}"' for k,v in kwargs.items()])+"\n"
    
    def get_role_end(self, role_name=None):

        if role_name == "system":
            return ""
        else:
            return "<|im_end|>\n"