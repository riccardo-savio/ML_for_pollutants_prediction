import pygame
import random
import sys

pygame.init()


bianco = (255, 255, 255)
verde = (20, 238, 27)
nero = (0, 0, 0)
larghezza_finestra = 800
altezza_finestra = 600

finestra = pygame.display.set_mode((larghezza_finestra, altezza_finestra))
pygame.display.set_caption(" Ping Pong - Ccr317")
clock = pygame.time.Clock()
FPS = 60

larghezza_racchetta = 15
altezza_racchetta = 100
velocita_racchetta = 10

racchetta_sinistra = pygame.Rect(
    50,
    altezza_finestra // 2 - altezza_racchetta // 2,
    larghezza_racchetta,
    altezza_racchetta,
)
racchetta_destra = pygame.Rect(
    larghezza_finestra - 50 - larghezza_racchetta,
    altezza_finestra // 2 - altezza_racchetta // 2,
    larghezza_racchetta,
    altezza_racchetta,
)

dimensione_palla = 20

velocita_palla_x = 5
velocita_palla_y = 5

palla = pygame.Rect(
    larghezza_finestra // 2 - dimensione_palla // 2,
    altezza_finestra // 2 - dimensione_palla // 2,
    dimensione_palla,
    dimensione_palla,
)

dimensione_blocco = 20
blocco = pygame.Rect(
    800 // 2 - dimensione_blocco // 2,
    600 // 4 - dimensione_blocco // 2,
    dimensione_blocco,
    dimensione_blocco,
)

dimensione_blocco2 = 20
blocco2 = pygame.Rect(
    800 // 2 - dimensione_blocco2 // 2,
    600 // 1.3 - dimensione_blocco2 // 2,
    dimensione_blocco2,
    dimensione_blocco2,
)

dimensione_blocco3 = 20
blocco3 = pygame.Rect(
    800 // 2 - dimensione_blocco3 // 2,
    600 // 2 - dimensione_blocco3 / 2,
    dimensione_blocco3,
    dimensione_blocco3,
)

punti2 = 0
punti = 0

font = pygame.font.SysFont("Arial", 60, bold=True, italic=False)


def msg(message, color):
    message = font.render(message, True, color)
    finestra.blit(message, [larghezza_finestra / 10, altezza_finestra / 3])


def disegna_oggetti():
    finestra.fill(nero)
    pygame.draw.rect(finestra, bianco, racchetta_sinistra)
    pygame.draw.rect(finestra, bianco, racchetta_destra)
    pygame.draw.ellipse(finestra, bianco, palla)
    pygame.draw.rect(finestra, verde, blocco)
    pygame.draw.rect(finestra, verde, blocco2)
    pygame.draw.rect(finestra, verde, blocco3)

    testo2 = font.render(str(punti2), True, bianco)
    testo = font.render(str(punti), True, bianco)

    finestra.blit(testo2, (larghezza_finestra // 2 - testo.get_width() + 325, 60))
    finestra.blit(testo, (larghezza_finestra // 2 - testo.get_width() - 325, 60))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    tasti = pygame.key.get_pressed()

    if tasti[pygame.K_w]:
        racchetta_sinistra.y -= velocita_racchetta
    if tasti[pygame.K_s]:
        racchetta_sinistra.y += velocita_racchetta
    if tasti[pygame.K_UP]:
        racchetta_destra.y -= velocita_racchetta
    if tasti[pygame.K_DOWN]:
        racchetta_destra.y += velocita_racchetta

    pygame.display.update()
    palla.x += velocita_palla_x
    palla.y += velocita_palla_y

    if palla.top <= 0 or palla.bottom >= altezza_finestra:
        velocita_palla_y = -velocita_palla_y

    if palla.left <= 0 or palla.right >= larghezza_finestra:
        palla.x = larghezza_finestra // 2 - dimensione_palla // 2
        palla.y = altezza_finestra // 2 - dimensione_palla // 2

        velocita_palla_x = random.choice([-5, 5])
        velocita_palla_y = random.choice([-5, 5])

        punti = 0
        pygame.display.update()
    if palla.colliderect(racchetta_sinistra):
        velocita_palla_x = -velocita_palla_x
        velocita_palla_x += 0.5
        velocita_palla_y += 0.5
        punti += 1
    if palla.colliderect(blocco):
        velocita_palla_x = -velocita_palla_x

    if palla.colliderect(blocco2):
        velocita_palla_x = -velocita_palla_x

    if palla.colliderect(blocco3):
        velocita_palla_x = -velocita_palla_x

    if palla.colliderect(racchetta_destra):
        velocita_palla_x = -velocita_palla_x
        velocita_palla_x += 0.5
        velocita_palla_y += 0.5
        punti2 += 1
    if punti == 2:
        msg("Racchetta_sinistra ha vinto", bianco)
        velocita_racchetta = 10
        velocita_palla_x = 5
        velocita_palla_y = 5
        punti = 0
        punti2 = 0
    if punti2 == 20:
        velocita_racchetta = 10
        velocita_palla_x = 5
        velocita_palla_y = 5
        punti = 0
        punti2 = 0
    if velocita_palla_y >= 10 and velocita_palla_x >= 10:
        velocita_racchetta == 15

    disegna_oggetti()

    pygame.display.update()
    clock.tick(FPS)
