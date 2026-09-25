"""Small synchronous client. Retrying a mutation preserves its request ID."""
import uuid
import httpx

class Client:
    def __init__(self,base_url,team_key,transport=None):
        self.http=httpx.Client(base_url=base_url,headers={'Authorization':'Bearer '+team_key},timeout=360,transport=transport)
    def close(self): self.http.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def _post(self,path,payload,request_id=None):
        headers={'Idempotency-Key':request_id or str(uuid.uuid4())}
        for attempt in range(2):
            try:
                response=self.http.post(path,json=payload,headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                if attempt: raise
    def reset(self,system_id,request_id=None): return self._post('/reset',{'system_id':system_id},request_id)
    def step(self,run_id,intervention,request_id=None): return self._post('/step',{'run_id':run_id,'intervention':intervention},request_id)
    def budget(self,system_id):
        response=self.http.get('/budget/'+system_id); response.raise_for_status(); return response.json()
    def brief(self,system_id):
        response=self.http.get('/brief/'+system_id); response.raise_for_status(); return response.json()
    def documents(self,system_id):
        response=self.http.get('/documents/'+system_id); response.raise_for_status(); return response.json()
