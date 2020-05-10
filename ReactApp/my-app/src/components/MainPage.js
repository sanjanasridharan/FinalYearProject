import React,{useState} from 'react'
import MenuAppBar from './MenuAppBar'
import ListOfSym from './ListOfSym'
import ComboBox from './EnterSymptom'
import axios from 'axios'
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';


function MainPage() {
    const t="Welcome Sanjana"
    const [displaySym,setSym]=useState(false)
    const [displayDis,setDis]=useState(false)
    const [symp,setSymptom] = useState('')
    const [SympList,setList]=useState([])
   
    const getSymptom = () => {
     console.log(symp)
     fetch('http://localhost:5000/login',{
      method: 'POST',
      headers : { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
       },
       body: JSON.stringify({
        'symptom': symp})

    }).then(res => res.json()).then(data => {
      console.log(data.time);
      setList(data.time)
    });
    setSym(true)
    }
      function getDisease() {     
        setDis(true)
       }
    return (
        <div>
             <MenuAppBar props={t}/>
             
             {/* <ComboBox/>
              */}
              <TextField variant="filled"  margin="normal" required fullWidth  label="Enter Symptom"  value={symp}  autoFocus onChange={e => setSymptom(e.target.value)} />
             <Button style={{marginTop:'20px',marginRight:'96px',width:'300px'}}variant="contained" color="primary" onClick={getSymptom}>
              Predict similar symptoms
             </Button>
           {displaySym ?
           <div>
               <div style={{marginTop:'20px',marginRight:'247px'}}> <h>Select the Symptoms:</h>
               <div style={{marginLeft:'240px'}}>
                <ListOfSym props={SympList}/>
               </div>
               </div>
               <Button style={{marginTop:'20px',marginRight:'96px',width:'300px'}}variant="contained" color="primary" onClick={getDisease}>
              Predict Disease
             </Button>
               {displayDis ?
               <div><div style={{marginTop:'20px',marginRight:'247px'}}> <h>The Predicted Disease:</h>
               <div style={{marginLeft:'180px',marginTop:'20px'}}>
               <TextField
          id="filled-read-only-input"
          label="Disease"
          defaultValue="Cancer"
          InputProps={{
            readOnly: true,
          }}
          variant="filled"
        />
 
               </div>
               </div></div>:
               <div></div>}
               </div>:
               
               <p></p>
            }
        </div>
    )
}

export default MainPage
