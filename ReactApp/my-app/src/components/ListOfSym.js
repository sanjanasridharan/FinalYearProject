import React from 'react';
import Checkbox from '@material-ui/core/Checkbox';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import Button from '@material-ui/core/Button';
import Avatar from '@material-ui/core/Avatar';
import TextField from '@material-ui/core/TextField';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import Autocomplete from '@material-ui/lab/Autocomplete';
import {CheckStatus} from '../context/CheckStatus'

export default function ListOfSym({props}) {
  const [arr,setArr]=React.useState([])
  const [value, setValue] = React.useState();
  const user=React.useContext(CheckStatus)
  const displaySym = (props) => {
    
    if(props.e.target.checked)
        arr.push(props.sym)
        
    if(!props.e.target.checked)
      {let i=arr.indexOf(props.sym)
        arr.splice(i,i)
      }
      console.log(arr)
      user.setDataValues(arr)
    };
  const additionalSym = (p) => {
    console.log(p.title)
    arr.push(p.title)
    console.log(arr)
    user.setDataValues(arr)
  };
  

    return (
      <div>
      <FormControl  component="fieldset">
        <FormGroup style={{padding:'20px'}} aria-label="position" row>
         {props.map((sym) =>(
          <FormControlLabel
            draggable
            value="end"
            control={<Checkbox color="primary"  />}
            label={sym}
            labelPlacement="end"
            onClick={(e => displaySym({e,sym}))}
            onDragEnd={(e => window.open("https://www.google.com/search?q="+sym, "_blank"))}
         />))} 
        </FormGroup>
      </FormControl>
        <Container style={{marginTop:'50px'}} component="main" maxWidth="xs">
        <Autocomplete
                    id="controllable-states-demo"
                    value={value}
                    onChange={(event, newValue) => {
                     setValue(newValue);
                             }}
                    options={top100Sym}
                    getOptionLabel={(option) => option.title}
                    // style={{ width: 300,marginTop:'120px',marginLeft:'600px'}}
                    renderInput={(params) => <TextField {...params} label="Enter Additional Symptom" variant="outlined" />}
    />
        <Button style={{marginTop:'25px'}} type="submit" fullWidth variant="contained" color="primary" onClick={(e=>additionalSym(value))}>+ ADD</Button>
          </Container>
      </div>
    );
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
  