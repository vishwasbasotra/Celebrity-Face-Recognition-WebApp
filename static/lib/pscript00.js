var playerName = document.getElementById("playerName").value;

$("#messi").hide();
$("#sharapova").hide();
$("#federer").hide();
$("#serena").hide();
$("#virat").hide()

if (playerName == "lionel_messi") {
    $("#messi").show();
}
if (playerName == "maria_sharapova") {
    $("#sharapova").show();
}
if (playerName == "roger_federer") {
    $("#federer").show();
}
if (playerName == "serena_williams") {
    $("#serena").show();
}
if (playerName == "virat_kohli") {
    $("#virat").show();
}