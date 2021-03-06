import React,{useState} from 'react'
import MenuAppBar from './MenuAppBar'
import ListOfSym from './ListOfSym'
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import {CheckStatus} from '../context/CheckStatus'
import SimpleModal from './SimpleModal'


function MainPage() {
    
    const user=React.useContext(CheckStatus)
    const name=user.username
    const t="Welcome "+name+""
    const [displaySym,setSym]=useState(false)
    const [displayDis,setDis]=useState(false)
    const [symp,setSymptom] = useState('')
    const [SympList,setList]=useState([])
    const [value, setValue] = React.useState();
    const [finalDis,setfinal]=useState('')
    const getSymptom = () => {
     console.log(symp)
     fetch('http://localhost:5000/login',{
      method: 'POST',
      headers : { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
       },
       body: JSON.stringify({
        'symptom': value.title})

    }).then(res => res.json()).then(data => {
      console.log(data.time);
      setList(data.time)
    });
    setSym(true)
    }


      function getDisease() {     
        fetch('http://localhost:5000/RandomForest',{
      method: 'POST',
      headers : { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
       },
       body: JSON.stringify({
        'symptom':user.symList})

    }).then(res => res.json()).then(data => {
      console.log(data.time);
      setfinal(data.time)
      user.setPredictedDis(data.time)
    });
        setDis(true)
       }
       
    return (
        <div>
             <MenuAppBar props={t}/>
              <Autocomplete
                    id="controllable-states-demo"
                    value={value}
                    onChange={(event, newValue) => {
                     setValue(newValue);
                             }}
                    options={top100Sym}
                    getOptionLabel={(option) => option.title}
                    style={{ width: 300,marginTop:'120px',marginLeft:'600px'}}
                    renderInput={(params) => <TextField {...params} label="Enter Symptom" variant="outlined" />}
    />
    
             <Button style={{marginTop:'20px',marginRight:'30px',width:'300px'}}variant="contained" color="primary" onClick={getSymptom}>
              Predict similar symptoms
             </Button>
           {displaySym ?
           <div>
               <div style={{marginTop:'20px',marginRight:'247px'}}> <h>Select the Symptoms:</h>
               <div style={{marginLeft:'240px'}}>
                <ListOfSym props={SympList}/>
               </div>
               </div>
               <Button style={{marginTop:'20px',marginRight:'30px',width:'300px'}}variant="contained" color="primary" onClick={getDisease}>
              Predict Disease
                </Button>
               {displayDis ?
               <div>
                 {/* <div style={{marginTop:'20px',marginRight:'247px'}}> <h>The Predicted Disease:</h>
               <div style={{marginLeft:'180px',marginTop:'20px'}}>
               <TextField
          id="filled-read-only-input"
          label={finalDis}
          defaultValue={finalDis}
          InputProps={{
            readOnly: true,
          }}
          variant="filled"
        />
 
               </div> */}
               <SimpleModal/>
               </div>:
               <div></div>}
               </div>:
               
               <p></p>
            }
        </div>
    )
}
const top100Sym = [
  { title: 'shortness of breath' },
  { title: 'pain' },
  { title: 'fever'},
  { title: 'pain abdominal'},
  { title: 'diarrhea'},
  { title: "vomiting" },
  { title: 'asthenia'},
  { title: 'cough' },
  { title: 'dyspnea'},
  { title: 'nausea'},
  { title: 'unresponsiveness' },
  { title: 'chill' },
  { title: 'pain chest'},
  { title: 'apyrexial' },
  { title: "decreased body weight" },
  { title: 'agitation'},
  { title: 'rale' },
  { title: 'lesion'},
  { title: 'mass of body structure'},
  { title: 'hypotension' },
  { title: 'sore to touch' },
  { title: 'hallucinations auditory'},
  { title: "night sweat" },
  { title: 'orthopnea' },
  { title: 'syncope'},
  { title: 'thicken'},
  { title: 'haemorrhage'},
  { title: 'swelling'},
  { title: 'tremor' },
  { title: 'distress respiratory'},
  { title: 'feeling suicidal' },
  { title: 'hypokinesia' },
  { title: 'patient non compliance' },
  { title: 'suicidal'},
  { title: 'feeling hopeless'},
  { title: 'irritable mood'},
  { title: 'sleepy'},
  { title: 'sweating increased'},
  { title: 'tachypnea'},
  { title: 'wheezingt' },
  { title: 'worry'},
  { title: 'ascites' },
  { title: 'blackout' },
  { title: 'difficulty' },
  { title: 'dyspnea on exertion' },
  { title: 'headache'},
  { title: 'hemiplegia' },
  { title: 'hyponatremia' },
  { title: 'non-productive cough' },
  { title: 'pleuritic pain'},
  { title: 'pruritus' },
  { title: 'seizure	'},
  { title: 'sleeplessness' },
  { title: 'angina pectoris' },
  { title: 'constipation' },
  { title: 'facial paresis'},
  { title: 'fall' },
  { title: 'fatigue'},
  { title: 'hallucinations visual'},
  { title: 'hemodynamically stable' },
  { title: 'hyperkalemia' },
  { title: 'mental status changes' },
  { title: 'palpitation'},
  { title: 'productive cough' },
  { title: 'anorexia' },
  { title: 'bradycardia' },
  { title: 'chest tightness' },
  { title: 'dizziness' },
  { title: 'guaiac positive' },
  { title: 'homelessness' },
  { title: 'prostatism' },
  { title: 'tumor cell invasion' },
  { title: 'abdominal tenderness' },
  { title: 'abscess bacterial' },
  { title: 'chest discomfort' },
  { title: 'consciousness clear'},
  { title: 'decreased translucency' },
  { title: 'distended abdomen' },
  { title: 'erythema' },
  { title: 'jugular venous distention'},
  { title: 'lethargy' },
  { title: 'mood depressed' },
  { title: 'myalgia' },
  { title: 'redness' },
  { title: 'rhonchus'},
  { title: 'transaminitis' },
  { title: 'unconscious state' },
  { title: 'unsteady gait' },
  { title: 'weepiness' },
  { title: 'breath sounds decreased'},
  { title: 'dysarthria'},
  { title: "hematuria"},
  { title: 'intoxication'},
  { title: 'muscle twitch'},
  { title: 'nightmare' },
  { title: 'numbness' },
  { title: 'pressure chest' },
  { title: 'sinus rhythm'},
  { title: 'yellow sputum'},
  { title: 'verbal auditory hallucinations'},
];

export default MainPage
