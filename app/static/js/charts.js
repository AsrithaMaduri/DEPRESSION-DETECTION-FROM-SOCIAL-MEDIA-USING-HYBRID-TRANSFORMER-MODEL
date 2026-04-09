const acc = document.getElementById('accuracyChart');

new Chart(acc, {

type: 'line',

data: {

labels: ["Epoch1","Epoch2","Epoch3","Epoch4","Epoch5"],

datasets: [{

label: "Training Accuracy",

data: [0.78,0.84,0.88,0.91,0.93],

borderWidth: 2

}]

}

});

const loss = document.getElementById('lossChart');

new Chart(loss, {

type: 'line',

data: {

labels: ["Epoch1","Epoch2","Epoch3","Epoch4","Epoch5"],

datasets: [{

label: "Training Loss",

data: [0.65,0.49,0.38,0.29,0.22],

borderWidth: 2

}]

}

});