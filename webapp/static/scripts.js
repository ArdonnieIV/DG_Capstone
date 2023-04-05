const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const webcam = new Webcam(webcamElement, 'user', canvasElement);


webcam.start()
   .then(result =>{
      console.log("webcam started");
   })
   .catch(err => {
        console.log("error")
       console.log(err);
   });

function takePic(){
    //let picture = webcam.snap();
    //webcam.stop();
    let picture = "here"
    sendPose(picture)
}

function sendPose(picture){
    let data = {"imgfile" : picture};
    fetch('/predict', {
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": JSON.stringify(data),})
      .then(function (response) {
          return response.json();
      }).then(function (text) {
          console.log('GET response:');
          console.log(text.pose_name); 
          //document.getElementById('pose-response').innerHTML = text.pose_name;
          //document.getElementById('pose-score').innerHTML = text.pose_score;
        });

}
