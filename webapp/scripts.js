const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const webcam = new Webcam(webcamElement, 'user', canvasElement);


webcam.start()
   .then(result =>{
      console.log("webcam started");
   })
   .catch(err => {
       console.log(err);
   });

   

function takePic(){
    let picture = webcam.snap();
    document.querySelector('#download-photo').href = picture;
    webcam.stop();
}