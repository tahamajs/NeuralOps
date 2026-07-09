export function getAuthToken(): string {
  const local = localStorage.getItem("token");
  if (local) return local;
  return document.cookie.split("; ").find((row) => row.startsWith("authToken="))?.split("=")[1] || "";
}

export function setAuthToken(token: string): void {
  localStorage.setItem("token", token);
  document.cookie = `authToken=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
}

export function removeAuthToken(): void {
  localStorage.removeItem("token");
  document.cookie = "authToken=; path=/; max-age=0; SameSite=Lax";
}
