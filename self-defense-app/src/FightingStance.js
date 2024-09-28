import React from "react";

const FightingStance = () => {
    return (
        <div>
            <h2>Fighting Stance Practice</h2>
            <img
                src="http://localhost:8000/video_feed"
                alt="Webcam feed"
                style={{ width: "640px", height: "480px" }}
            />
        </div>
    );
};

export default FightingStance;
