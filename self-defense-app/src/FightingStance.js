import React from "react";

const FightingStance = () => {
    return (
        <div>
            <h2>Fighting Stance Practice</h2>
            <iframe
                src="http://localhost:8501"
                title="Webcam feed"
                style={{ width: "640px", height: "480px", border: "none" }}
            />
        </div>
    );
};

export default FightingStance;
